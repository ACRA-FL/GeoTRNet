"""
Traing GeoTR model on COCO-annotated data in general
"""

import os
import argparse
import math
import tensorflow as tf
import numpy as np
from model import GeoTRNet, mAP, mDP
from digitgen import DigitGenerator

class DigitGenDataset(object):
    def __init__(self,gen:DigitGenerator):
        self.gen = gen
        self.category_map = None

    def gen_label_array(self,ret_ann):
        category_map = ret_ann["category"]
        label_shape = (self.gen.samples,self.gen.digit_size)
        label_array = np.zeros(label_shape)
        for sample_id in range(self.gen.samples):
            for digit_id in range(self.gen.digit_size):
                ann = ret_ann["annotations"][sample_id+digit_id]
                label_array[sample_id,digit_id] = ann["category_id"]

        return label_array,category_map

    def gen_data(self):
        self.gen.generate_digits()
        ret_arr, ret_ann = self.gen.generate_dataset()

        ret_ann, category_map = self.gen_label_array(ret_ann)
        return ret_arr, ret_ann, category_map

    def generate(self):
        init_image, init_label,category_map = self.gen_data()
        self.category_map = category_map

        ret_tensor = tf.convert_to_tensor(init_image, dtype=tf.float32)
        ind_tensor = tf.convert_to_tensor(init_label, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((ret_tensor, ind_tensor))


def train(cfg):

	train_gen = DigitGenerator(8,samples=cfg.n_training,image_size=cfg.image_size,gray_scale=True,transpose=True)
	test_gen = DigitGenerator(8,samples=cfg.n_validation,image_size=cfg.image_size,gray_scale=True,transpose=True)
	gen_ds_train = DigitGenDataset(train_gen)
	gen_ds_test = DigitGenDataset(test_gen)

	train_dataset = gen_ds_train.generate()
	test_dataset = gen_ds_test.generate()
	train_loader = train_dataset.shuffle(buffer_size=1000).batch(cfg.bs, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
	valid_loader = test_dataset.batch(cfg.bs, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

	def lr_step_decay(epoch, lr):
		drop_rate = cfg.lr_drop_rate
		epochs_drop = cfg.lr_patient
		return lr * math.pow(drop_rate, math.floor(epoch/epochs_drop))

	lr_decaying_callback = tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)

	earlystopping_callback = tf.keras.callbacks.EarlyStopping(
                            monitor=cfg.es_metrics, #'val_accuracy',
                            min_delta=0,
                            patience=cfg.es_patientce,
                            restore_best_weights=True
                        )
	optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.starting_lr)

	model = GeoTRNet(input_size=cfg.image_size,
					bs=cfg.bs,
					n_classes=cfg.n_classes, #10,
					max_obj=cfg.max_objects,
					h_cell=cfg.h_cell,
					latent_size=cfg.latent_size,
					linear_k=cfg.linear_k,
					with_TCN=cfg.with_CTN,
					with_SAM=cfg.with_SAM,
					rho=cfg.rho)

	model.compile(optimizer, loss='categorical_crossentropy', run_eagerly=True)

	model.fit(
		train_loader,
		epochs=cfg.n_epochs,
		validation_data=valid_loader,
		callbacks=[earlystopping_callback,
				lr_decaying_callback]
	)

	model.save(cfg.save_dir)
	