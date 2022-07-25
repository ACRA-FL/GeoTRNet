
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras import losses
from tcn import TCN

class mAP(tf.keras.metrics.Metric):
	def __init__(self, name='mAP', alpha=0.9, **kwargs):
		super(mAP, self).__init__(name=name, **kwargs)
		self.map = self.add_weight(name='map', initializer='zeros')
		self.alpha = alpha

	def update_state(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.int32)
		y_pred = tf.cast(y_pred, tf.float32)
		alpha = tf.constant(self.alpha, dtype=tf.float32)
		alpha_prime = tf.constant(1-self.alpha, dtype=tf.float32)

		res_arr = tf.map_fn(lambda i: tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true[i, ...], y_pred[i, ...])), elems=tf.range(y_pred.shape[0]), dtype=tf.float32)
		tensor_without_nans = tf.where(tf.math.is_nan(res_arr), tf.zeros_like(res_arr), res_arr)
		cur_val = tf.math.reduce_mean(tensor_without_nans, axis=-1)
		weighted_avg = tf.math.add(tf.multiply(alpha, self.map), tf.multiply(alpha_prime, cur_val))
		self.map.assign(weighted_avg)

	def result(self):
		return self.map

class mDP(tf.keras.metrics.Metric):
	def __init__(self, name='mDP', n_classes=10, alpha=0.9, **kwargs):
		super(mDP, self).__init__(name=name, **kwargs)
		self.n = n_classes
		self.alpha = alpha
		self.mdp = self.add_weight(name='mdp', initializer='zeros')

	def update_state(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.int32)
		y_pred = tf.cast(y_pred, tf.float32)
		alpha = tf.constant(self.alpha, dtype=tf.float32)
		alpha_prime = tf.constant(1-self.alpha, dtype=tf.float32)

		def func(i):
			mask = tf.equal(y_true, tf.constant(i))
			ilabels = tf.reshape(tf.boolean_mask(y_true,mask), shape=(-1))
			ipreds = tf.reshape(tf.boolean_mask(y_pred,mask), shape=(-1, self.n))
			return tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(ilabels, ipreds))

		res_arr = tf.map_fn(lambda i: func(i), elems=tf.range(self.n), dtype=tf.float32)
		tensor_without_nans = tf.where(tf.math.is_nan(res_arr), tf.zeros_like(res_arr), res_arr)
		cur_val = tf.math.reduce_mean(tensor_without_nans, axis=-1)
		weighted_avg = tf.math.add(tf.multiply(alpha, self.mdp), tf.multiply(alpha_prime, cur_val))
		self.mdp.assign(weighted_avg)

	def result(self):
		return self.mdp

class GeoTRNet(tf.keras.Model):
	def __init__(self, bs=16, input_size=(128, 48), n_classes=10, max_obj=15, h_cell=64, latent_size=32, linear_k=3, with_TCN=False, with_SAM=False, rho=0.05):
		super().__init__()
		self.rho = rho
		self.bs = bs
		self.input_size = input_size
		self.n_classes = n_classes  # LSTM module projection size
		self.h_cell = h_cell  # LSTM hidden size
		self.latent_size = latent_size
		self.max_obj = max_obj
		self.linear_k = linear_k
		self.with_SAM = with_SAM
		self.with_TCN = with_TCN
		self.model = self.build_model(
			self.input_size, self.h_cell, self.latent_size, self.n_classes, self.max_obj, self.linear_k, self.with_TCN)

		self.scce = losses.SparseCategoricalCrossentropy(from_logits=False,)

		self.loss_metric = tf.keras.metrics.Mean(name="loss")
		self.acc_metric = tf.keras.metrics.Mean(name="accuracy")
		self.mAP = mAP()
		self.mDP = mDP(n_classes=self.n_classes)
		self.acc_mat = tf.keras.metrics.Accuracy()


	def build_model(self, input_shape=(128, 48), h_cell=32, latent_size=32, n_classes=10, n_obj=10, linear_k=3, with_TCN=False):
		x_input = layers.Input(shape=input_shape)
		if with_TCN:
			x = TCN(nb_filters=h_cell, kernel_size=3,
				nb_stacks=1,
				dilations=(1, 2, 4, 8, 16),
				padding='same',
				use_skip_connections=True,
				dropout_rate=0.05,
				return_sequences=True,
				activation='tanh',)(x_input)
		else:
			forward_layer = layers.LSTM(h_cell//2, return_sequences=True)
			x = layers.Bidirectional(forward_layer, input_shape=self.input_size)(x_input)
			x = layers.LSTM(latent_size, activation='tanh',
							return_sequences=True, go_backwards=True)(x)
		x = layers.Conv1D(n_classes, linear_k, activation='tanh')(x)
		x = layers.Permute((2, 1))(x)
		x = layers.Conv1D(n_obj, 1, activation='tanh')(x)
		x = layers.Permute((2, 1))(x)
		x_output = layers.Activation('softmax')(x)

		model = tf.keras.Model(inputs=x_input, outputs=x_output)
		return model

	@property
	def metrics(self):
		return [self.loss_metric, self.acc_mat]

	def calculate_loss(self, target, pred):
		target = tf.reshape(target, shape=(-1))
		pred = tf.reshape(pred, shape=(-1, self.n_classes))
		loss = self.scce(target, pred)

		y_hat = tf.argmax(pred, axis=-1)
		acc = self.acc_mat(y_hat, target)
		return loss, acc

	def call(self, x):
		full_out = self.model(x)
		return full_out

	def train_step(self, batch_data):
		x, target = batch_data
		with tf.GradientTape() as tape:
			pred = self(x, training=True)
			# print(pred.shape, pred)
			loss, acc = self.calculate_loss(target, pred)

		if self.with_SAM:
			e_ws = []
			trainable_params = self.trainable_variables
			gradients = tape.gradient(loss, trainable_params)
			grad_norm = self._grad_norm(gradients)
			scale = self.rho / (grad_norm + 1e-12)

			for (grad, param) in zip(gradients, trainable_params):
				e_w = grad*scale 
				param.assign_add(e_w)
				e_ws.append(e_w)

			with tf.GradientTape() as tape:
				pred = self(x, training=True)
				loss, acc = self.calculate_loss(target, pred)

			gradient = tape.gradient(loss, trainable_params)
			for (param, e_w) in zip(trainable_params, e_ws):
				param.assign_sub(e_w)
		else:
			gradient = tape.gradient(loss, self.trainable_variables)
			
		self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
		self.loss_metric.update_state(loss)
		self.acc_metric.update_state(acc)
		self.mAP.update_state(target, pred)
		self.mDP.update_state(target, pred)
		return {
			"loss": self.loss_metric.result(),
			"accuracy": self.acc_mat.result(),
			"mAP": self.mAP.result(),
			"mDP": self.mDP.result()
		}

	def test_step(self, batch_data):
		x, target = batch_data

		pred = self(x, training=False)
		loss, acc = self.calculate_loss(target, pred)

		self.loss_metric.update_state(loss)
		self.acc_metric.update_state(acc)
		self.mAP.update_state(target, pred)
		self.mDP.update_state(target, pred)
		return {
			"loss": self.loss_metric.result(),
			"accuracy": self.acc_mat.result(),
			"mAP": self.mAP.result(),
			"mDP": self.mDP.result()
		}
	
	def _grad_norm(self, gradients):
		norm = tf.norm(
			tf.stack([
				tf.norm(grad) for grad in gradients if grad is not None
			])
		)
		return norm