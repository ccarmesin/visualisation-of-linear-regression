import * as tf from '@tensorflow/tfjs';

export class Model {

    constructor() {

        // The model
        this.model = tf.sequential();

        // Hyperparameters
        this.LEARNING_RATE = .2;
        this.BATCH_SIZE = 10;
        this.EPOCHS = 80;

        // Optimizer
        this.optimizer = tf.train.sgd(this.LEARNING_RATE);

        // Line parameters
        this.m = tf.variable(tf.scalar(Math.random()));
        this.b = tf.variable(tf.scalar(Math.random()));

    }

    /**
     *
     * Train the model
     *
     * @param data to train the network on
     * @param log function to log current training state in ui
     * @param drawGraph function to draw current training state in a graph
     * 
     */
    async train(data, log, drawGraph) {

        // For more information about this see https://js.tensorflow.org/api/latest/#Training-Optimizers
        for (let i = 0; i < this.EPOCHS; i++) {
            const loss = this.optimizer.minimize(() => {

                const batch = data.nextTrainBatch(this.BATCH_SIZE);
                return this.loss(this.predict(batch.ys), batch.ys);

            }, true);

            // Log status and draw the graph
            log(loss.dataSync(), i, this.m, this.b);
            drawGraph(this.m, this.b);
            
            await tf.nextFrame();
        }

    }

    /**
     *
     * Make predictions on the model
     * Make sure you have train it before:)
     *
     * @param xs are values to make the prediction on
     * 
     * @return predictions of the model
     * 
     */
    predict(xs) {
        const pred = tf.tidy(() => {
            return xs.mul(this.m).add(this.b);
        });
        return pred;
    }

    /**
     *
     * Loss function(MSE in this case)
     *
     * @param predictions
     * @param labels
     * 
     * @return error
     * 
     */
    loss(predictions, labels) {

        return predictions.sub(labels).square().mean();

    }

}
