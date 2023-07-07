import tensorflow as tf
import numpy as np
import os
import time

# custimez loss here
class wL2(tf.keras.losses.Loss):
    def __init__(self, LOSS_GAMMA = 1.0):
        super().__init__()
        self.gamma = LOSS_GAMMA

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        y_true = tf.squeeze(y_true)
        se = tf.math.square(y_pred-y_true)
        wse = self.gamma*se + (1-self.gamma)*tf.math.multiply(se, y_true)
        wmse = tf.reduce_mean(wse)
        return wmse

# training
class Train():
    def __init__(self, model, save_folder_name:str, x_n:int, x_m:int, dataset, grad_cam) -> None:
       self.model = model
       self.save_folder_name = save_folder_name
       self.x_n = x_n
       self.x_m = x_m
       self.dataset = dataset
       self.grad_cam = grad_cam
       self.grad_cam_folder = os.path.join(self.save_folder_name, 'pred_epoch')

    def create_dirs(self):
        if os.path.exists(self.save_folder_name) is not True: os.mkdir(self.save_folder_name)
        if os.path.exists(self.grad_cam_folder) is not True: os.mkdir(self.grad_cam_folder)
        return True

    def start_train(self, optimizer, loss_fn, train_acc_metric, val_acc_metric,
                    epochs:int, save_pred_every_epoch=50, patience=40, lr_scheduler=False):
        '''
        # Input
            optimizer: tf.keras.optimizers class, defined in YourModel.py
            loss_fn: tf.keras.losses class, defined in YourModel.py
            train_acc_metric:  tf.keras.metrics class , defined in YourModel.py
            val_acc_metric: tf.keras.metrics class, defined in YourModel.py
            save_pred_every_epoch: the number of epochs during training
                            for saving the intermediate prediction
                            based on self.grad_cam defined by yourself.
            patience: the early stopping setting.
        '''
        # model compile
        print("Model Compiling ... ")
        if(lr_scheduler):
            lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1e4, 1e5, 1e6], [1e-2, 1e-3, 1e-4, 1e-5])
            optimizer = tf.keras.optimizers.Adam(lr)
        # self.model.compile(optimizer=optimizer, loss=loss_fn)
        print("Complete ... ")

        training_metrics = []
        val_metrics = []
        time_history = []
        lr = []

        # early stopping constants
        wait = 0
        best = float('inf')

        print("Creating Directory ... ")
        self.create_dirs()
        print("Complete ... ")

        print("Starting Training ... ")
        # total timer
        total_start_time = time.time()

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            train_acc_metric.update_state(y, logits)

        @tf.function
        def val_step(x, y):
            val_logits = self.model(x, training=False)
            val_acc_metric.update_state(y, val_logits) # Update val metrics

        for num_epochs in range(epochs):
            # epoch timer
            start_time = time.time()

            # Iterate over the batches of the training dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.dataset.train_dataset_gen()): 
                loss_value = train_step(x_batch_train, y_batch_train)

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            training_metrics.append(float(train_acc))
            train_acc_metric.reset_states() # Reset training metrics at the end of each epoch

            # batching through test dataset
            # for num_test_batches in range(int(TOTAL_TEST_NUM / BATCH_SIZE)):
            for step, (x_batch_val, y_batch_val) in enumerate(self.dataset.val_dataset_gen()):
                val_step(x_batch_val, y_batch_val)

            val_acc = val_acc_metric.result()
            val_metrics.append(float(val_acc))
            val_acc_metric.reset_states()
            
            # record spent time for each epoch
            time_history.append(float(time.time() - start_time))
            # lr.append(optimizer.learning_rate.numpy())
            print(" epochs: ", num_epochs+1, f"/{epochs}",
                    "\n, train metric: ", "{:.2e}".format(float(train_acc)),
                    "\n, val metric: ", "{:.2e}".format(float(val_acc)),
                    "\n, time: ", "{:.4f}".format(float(time.time() - start_time)), " s.")

            # save the intermediate output
            if(num_epochs % save_pred_every_epoch == 0):
                # save the output during training
                inter_output_by_epoch = self.model.predict(self.grad_cam)
                np.save(self.grad_cam_folder + f"/pred_epoch{num_epochs}", inter_output_by_epoch)
            # The early stopping strategy: stop the training if 'val_acc' does not
            # decrease over a certain number of epochs.
            wait += 1
            if(val_acc < best):
                best = val_acc
                wait = 0
            if(wait >= patience):
                break


        print("Training Ends. Spent total time = {:.2f}".format(float(time.time() - total_start_time)), " s.")

        print("Saving ...")
        # save the weights of the model layers
        self.model.save_weights(self.save_folder_name + f"/variables")
        # save losses
        loss_plot = np.array(training_metrics)
        val_loss_plot = np.array(val_metrics)
        time_history = np.array(time_history)
        # lr = np.array(lr)
        np.save(os.path.join(self.save_folder_name,  'losses.npy'), loss_plot)
        np.save(os.path.join(self.save_folder_name, 'val_losses.npy'), val_loss_plot)
        np.save(os.path.join(self.save_folder_name, 'time_history.npy'), time_history)
        # np.save(self.save_folder_name + "/lr.npy", lr)
        print("Saving Complete ... ")

if __name__ == '__main__' :
    print("Test Module OK.")
