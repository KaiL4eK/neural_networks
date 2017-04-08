
def preprocess_regress(imgs):
	imgs_p  = np.ndarray((imgs.shape[0],  img_side_size, img_side_size, 3), dtype=np.uint8)
	for i in range(imgs.shape[0]):
		imgs_p[i]  = cv2.resize(imgs[i],  (img_side_size, img_side_size), interpolation = cv2.INTER_NEAREST)

	return imgs_p

debug = True

def train_regression():
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, imgs_bbox_train = load_train_data()
	imgs_train = preprocess_regress(imgs_train)

	imgs_train = imgs_train.astype('float32') 
	imgs_train -= mean
	imgs_train /= 255.

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model = regression_model()
	# model.load_weights('last_weights.h5')

	print('-'*30)
	print('Fitting model...')
	print('-'*30)

	if not debug:
		model.fit(imgs_train, imgs_bbox_train, batch_size=32, epochs=1000, verbose=1, shuffle=True,
				  validation_split=0.11, callbacks=[ModelCheckpoint('weights.h5', monitor='loss', save_best_only=True)])
	else:
		while (True):
			model.fit(imgs_train, imgs_bbox_train, batch_size=32, epochs=1, verbose=1, shuffle=True, validation_split=0.11)
			model.save_weights('weights.h5')

			images = os.listdir(data_path)
			total = len(images)
			i_image = int(random.random() * total)

			test_img = cv2.imread(os.path.join(data_path, images[i_image]), cv2.IMREAD_COLOR)
			cv2.imshow('1', test_img)
			test_img = cv2.resize(test_img, (img_side_size, img_side_size), interpolation = cv2.INTER_NEAREST)
			test_img = test_img.astype('float32')
			test_img -= mean
			test_img /= 255.
			
			imgs_mask = model.predict(np.reshape(test_img, (1, img_side_size, img_side_size, 3)), verbose=1)

			if cv2.waitKey(100) == 27:
				cv2.destroyAllWindows()
				exit(1)
				