import * as tfvis from '@tensorflow/tfjs-vis';
import * as vis from 'vis';

import {
    GoogleCharts
} from 'google-charts';


const statusElement = document.getElementById('status'),
    messageElement = document.getElementById('message'),
    predictionsElement = document.getElementById('predictions'),
    gradientDiv = document.getElementById('gradient_div'),
    overlayDiv = document.getElementById('overlay'),
    chartRect = document.getElementsByTagName('rect');

const lossArr = [];

/**
 *
 * Draw the start of training in the ui
 * 
 */
export async function isTraining() {
    statusElement.innerText = 'Training...';
}

/**
 *
 * Draw the current training status to the ui
 *
 * @param loss
 * @param iteration
 * @param m slope
 * @param b y-intercept
 * 
 */
export async function trainingLog(loss, iteration, m, b) {

    messageElement.innerText = `\nloss[${iteration}]: ${loss} \n m: ${m} \n b: ${b}`;

}

/**
 *
 * Calculate the difference between labels and predictions and draw then to the ui
 *
 * @param predictionsT predictions of the network as Tensor
 * @param labelsT labels to the predictions as Tensor
 * 
 */
export async function showTestResults(predictionsT, labelsT) {

    statusElement.innerText = 'Testing...';

    let predictions = Array.from(predictionsT.dataSync());
    let labels = Array.from(labelsT.dataSync());

    for (let i = 0; i < predictions.length; i++) {

        // Calculate the difference between prediction and label
        const prediction = predictions[i].toFixed(1),
            label = labels[i].toFixed(1),
            difference = Math.abs(predictions[i] - labels[i]).toFixed(1);

        // Form ui element
        const pred = document.createElement('div');
        pred.className = 'column-sm card card-body';
        pred.innerHTML = `pred: ${prediction}<br>label: ${label}<br>diff: ${difference}`;

        predictionsElement.appendChild(pred);

    }

    statusElement.innerText = 'Finished';

}

/**
 *
 * Show n datapoints of the trainingdata and the function that tries to approximate this datapoints as best as possible
 * This all is shown in a Google Scatter Chart
 *
 * @param data class of Data containing all training and test examples
 * @param n number of datapoints to draw to the screen
 * 
 */
export async function showFunction(data, n) {

    //Load the charts library with a callback
    GoogleCharts.load(() => {
        let batch = data.nextTrainBatch(n);

        const array = [];
        const xs = Array.from(batch.xs.dataSync());
        const ys = Array.from(batch.ys.dataSync());

        array.push(['x', 'y']);
        for (let i = 0; i < xs.length; i++) {
            let dataObject = [xs[i], ys[i]];
            array.push(dataObject);
        }

        let dataArray = google.visualization.arrayToDataTable(array);

        let options = {
            legend: 'none',
            width: 500,
            height: 500,
            hAxis: {
                title: 'x',
                viewWindow: {
                    min: 0,
                    max: 1
                }
            },
            vAxis: {
                title: 'y',
                viewWindow: {
                    min: 0,
                    max: 1
                }
            },
            trendlines: {
                0: {
                    type: 'linear',
                    color: '#111',
                    opacity: .3
                }
            }
        };

        var chart = new google.visualization.ScatterChart(document.getElementById('chart_div'));

        chart.draw(dataArray, options);
    });

}

/**
 *
 * Draw a guessed line of the network
 * This line will more and more fit to the given line when training started
 *
 * @param mScalar slope as Tensor(Scalar)
 * @param bScalar intercept as Tensor(Scalar)
 * 
 */
export async function drawGuessedLine(mScalar, bScalar) {

    // Definition of a simple Line
    //y = m * x + b

    let width = overlayDiv.width;

    // Just convert Scalar to a normal Integer
    const mArray = Array.from(mScalar.dataSync());
    const bArray = Array.from(bScalar.dataSync());

    const m = mArray[0];
    const b = bArray[0];

    // Calculate coordinates for the line
    let m1 = 1 - m,
        x1 = 0,
        y1 = width - b * width,
        x2 = width,
        y2 = m1 * width - (b * width);

    // Draw the line into canvas
    let ctx = overlayDiv.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

}

/**
 *
 * Draw a 3dGraph of the loss function for the given function
 *
 * @param point contains click information(See http://visjs.org/docs/graph3d/)
 * 
 */
export async function drawLoss() {

    // Create and populate a data table.
    let data = new vis.DataSet();

    const steps = 20; // Number of datapoints will be steps*steps, if higher you will see more
    const axisMax = 1;
    const axisStep = axisMax / steps;

    for (let m = 0; m < axisMax; m += axisStep) {

        for (let b = 0; b < axisMax; b += axisStep) {

            // Calculate the error for these values
            let error = meanSquaredError(m, b);
            data.add({
                x: m,
                y: b,
                z: error
            });

        }

    }

    // Specify options
    let options = {
        onclick: onGraphClick,
        xLabel: 'm',
        yLabel: 'b',
        zLabel: 'error',
        style: 'surface',
        showPerspective: true,
        showGrid: true,
        showShadow: false,
        keepAspectRatio: true,
        verticalRatio: 0.5
    };

    // Create a graph3d
    new vis.Graph3d(gradientDiv, data, options);

}

/**
 *
 * Calculate the meanSquaredError up to xMax in steps
 *
 * @param m slope of the line
 * @param b intercept of the line
 * 
 */
function meanSquaredError(m, b) {

    let meanSquared = 0,
        xMax = 1,
        steps = .1,
        n = xMax / steps;

    for (let x = 0; x <= xMax; x += steps) {

        // Calculate y
        let guessY = m * x + b,
            labelY = x;

        // Calculate error for this step
        let error = guessY - labelY;
        error *= error;
        meanSquared += error;

        // Increase n(counter)
        n++;

    }

    // Divide summed error by the number of operations to get the mean
    return meanSquared /= n;

}

/**
 *
 * Handles a click event in a 3dGraph
 *
 * @param point contains click information(See http://visjs.org/docs/graph3d/)
 * 
 */
function onGraphClick(point) {

    let m = point.x.toFixed(1),
        b = point.y.toFixed(1),
        error = point.z.toFixed(1);

}
