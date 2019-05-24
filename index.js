import {
    Data
} from './data';
import {
    Model
} from './model';
import * as ui from './ui';

const data = new Data(),
      model = new Model();


main();
async function main() {

    await generate(.1);
    await train();
    await evaluate();
    ui.showFunction(data, 100);
    ui.drawLoss();

}

/**
 *
 * Generate data with a given noise and shuffle them
 *
 * @param noise, noise in the datapoints(between .01 and .2)
 * 
 */
async function generate(noise) {

    await data.generate(1, 0, noise);
    await data.shuffle();

}

/**
 *
 * Train the network and log the training
 * 
 */
async function train() {

    ui.isTraining();
    await model.train(data, ui.trainingLog, ui.drawGuessedLine);

}

/**
 *
 * Evaluate the model and show the results
 * 
 */
async function evaluate() {

    const batch = data.nextTestBatch(10);
    
    const labels = batch.ys,
          predictions = model.predict(batch.xs);

    ui.showTestResults(predictions, labels);

}