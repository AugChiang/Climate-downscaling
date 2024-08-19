import tensorflow as tf
import numpy as np
import os
import time

# training
class Train():
    def __init__(self,
                 model,
                 dataset,
                 config:object,
                 grad_cam = None
                ) -> None:

       self.model = model
       self.dataset = dataset
       self.config = config
       self.grad_cam = None

    def _check_dir(self):
        if os.path.exists(self.config.model_save_dir) is not True: os.mkdir(self.config.model_save_dir)
        return True

    def start_train(self,
                    optimizer,
                    loss_fn,
                    train_acc_metric,
                    val_acc_metric
                    ):
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
        training_metrics = []
        val_metrics = []
        time_history = []
        lr = []

        # early stopping constants
        wait = 0
        best = float('inf')

        print("Creating Directory ... ")
        self._check_dir()
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

        for num_epochs in range(self.config.epochs):
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
            print(" Epochs: ", num_epochs+1, f"/{self.config.epochs}",
                  "\n Train metric: ", "{:.2e}".format(float(train_acc)),
                  "\n Val metric: ", "{:.2e}".format(float(val_acc)),
                  "\n Time: ", "{:.4f}".format(float(time.time() - start_time)), " s.")
            print("=" * 40)

            # save the intermediate output
            if(self.grad_cam is not None):
                if(num_epochs % self.save_every_n_epoch == 0):
                    # save the output during training
                    inter_output_by_epoch = self.model.predict(self.grad_cam)
                    np.save(os.path.join(self.config.inter_pred_dir, f"pred_epoch{num_epochs}"), inter_output_by_epoch)
            # The early stopping strategy: stop the training if 'val_acc' does not
            # decrease over a certain number of epochs.
            wait += 1
            if(val_acc < best):
                best = val_acc
                wait = 0
            if(wait >= self.config.patience):
                break


        print("Training Ends. Spent total time = {:.2f}".format(float(time.time() - total_start_time)), " s.")

        print("Saving ...")
        # save the weights of the model layers
        self.model.save_weights(os.path.join(self.config.model_save_dir, "variables"))
        # save losses
        loss_plot = np.array(training_metrics)
        val_loss_plot = np.array(val_metrics)
        time_history = np.array(time_history)
        # lr = np.array(lr)
        np.save(os.path.join(self.config.history_save_dir, "losses.npy"), loss_plot)
        np.save(os.path.join(self.config.history_save_dir, "val_losses.npy"), val_loss_plot)
        np.save(os.path.join(self.config.history_save_dir, "time_history.npy"), time_history)
        # np.save(self.dir_name + "/lr.npy", lr)
        print("Saving Complete ... ")

if __name__ == '__main__' :
    print("Test Module OK.")
