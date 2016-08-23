#!/usr/bin/env python

import sys

from random import random, randint, choice
import numpy
import keras
from keras.models import Sequential, model_from_json, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD

def string_to_vector(input_string, length=255):
	input_string = input_string.lower()
	out_vector = numpy.zeros(shape=(length*26,), dtype=numpy.float) # 26 letters ONLY.  Space = all zeros.
	for index,character in enumerate(input_string):
		if index >= length:
			break
		if character.isalpha():
			out_vector[index*26 + ord(character)-ord('a')] = 1.0
	return out_vector

def vector_to_string(vector, threshold=0.1):
	output_string = ""
	for segment in range(0, vector.shape[0], 26):
		energy = random() # 0-1
		for i in range(0, 26):
			if energy < vector[segment+i]:
				output_string += chr(i + ord('a'))
				energy = 0
				break
			else:
				energy -= vector[segment+i]
		if energy > threshold:
			output_string += " " # Space when we've got leftover energy becayse nothing was selected.
	return output_string

def build_autoencoder():
	#model = Sequential()
	#model.add(Dense(input_dim=255*26, output_dim=512))
	#model.compile(loss='mean_absolute_error', optimizer='sgd')#, metrics=['accuracy']) # Try categorical_crossentropy

	encoder_input = Input(shape=(255*26,))
	lay = Dense(512, activation="relu")(encoder_input) # 1
	lay = Dense(128, activation="relu")(lay) # 2
	lay = Dense(512, activation="relu")(lay) # 3
	decoder = Dense(255*26, activation="relu")(lay)

	model = Model(input=encoder_input, output=decoder)
	model.compile(loss='binary_crossentropy', optimizer='adadelta')

	return model

def split_autoencoder(model):
	encoder_alt = Input(shape=(255*26,))
	encoder = Dense(512, weights=model.layers[1].get_weights())(encoder_alt)
	encoder = Dense(128, weights=model.layers[2].get_weights())(encoder)
	encoder_model = Model(input=encoder_alt, output=encoder)

	decoder_alt = Input(shape=(128,))
	decoder = Dense(512, weights=model.layers[3].get_weights())(decoder_alt)
	decoder = Dense(255*26, weights=model.layers[4].get_weights())(decoder)
	decoder_model = Model(input=decoder_alt, output=decoder)

	return encoder_model, decoder_model

def save_model(model, name):
	with open(name + "_structure.json", 'w') as fout:
		fout.write(model.to_json())
	model.save_weights(name + "_weights.h5")

def load_model(name):
	with open(name + "_structure.json", 'r') as fin:
		model = model_from_json(fin.read())
	model.load_weights(name + "_weights.h5")
	return model

def main():
	# Build training dataset
	fin = open("./data/sentences.txt", 'r')
	lines = fin.readlines()
	fin.close()
	data = numpy.zeros(shape=(100, string_to_vector("").shape[0]), dtype=numpy.float)
	for index,line in enumerate(lines):
		data[index,:] = string_to_vector(line)[:]
		if index >= 99:
			break
	
	# Build Mode.
	model = build_autoencoder()

	# Train
	print("Training model.")
	for i in range(1):
		print("Iteration {}".format(i))
		model.fit(data, data, shuffle=True)
		encoder, decoder = split_autoencoder(model)
		sample = choice(lines)
		print("Sample: {}".format(sample))
		enc = encoder.predict(numpy.atleast_2d(string_to_vector(sample)))
		#print("Enc: {}".format(enc))
		dec = decoder.predict(enc)
		print("Dec: {}".format(vector_to_string(dec[0,:])))

	# Save result
	save_model(model, "autoencoder")
	save_model(encoder, "encoder")
	save_model(decoder, "decoder")

if __name__=="__main__":
	main()
