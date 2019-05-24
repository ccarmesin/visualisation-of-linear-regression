/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

const NUM_DATASET_ELEMENTS = 600;

const TRAIN_TEST_RATIO = 4 / 5;

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

export class Data {

    constructor() {

        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;

    }

    /**
     *
     * Generate training- and testdata for given parameters with noise
     *
     * @param m slope
     * @param b y-intercept
     * @param noise of data
     * 
     */
    async generate(m, b, noise) {

        let randomX, randomY, noiseX, noiseY, dataset = [];
        // Generate random points near the given line 
        for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {

            randomX = Math.random();
            randomY = m * randomX + b;

            // Create given noise
            noiseX = (Math.random() * 2 - 1) * noise + 1;
            noiseY = (Math.random() * 2 - 1) * noise + 1;

            randomX = randomX * noiseX;
            randomY = randomY * noiseY;

            dataset.push({
                x: randomX * noiseX,
                y: randomY * noiseY
            });

        }

        this.trainDataset = dataset.slice(0, NUM_TRAIN_ELEMENTS);
        this.testDataset = dataset.slice(NUM_TRAIN_ELEMENTS);

    }

    /**
     *
     * Shuffle the dataset
     * 
     */
    async shuffle() {

        let shuffledTrainDataset = [],
            shuffledTestDataset = [];

        // Shuffle the dataset in packs of 1000 and push to shuffledTrainDataset
        await tf.data.array(this.trainDataset).shuffle(1000).forEach(app => shuffledTrainDataset.push(app));
        this.trainDataset = shuffledTrainDataset;

        // Same with test dataset
        await tf.data.array(this.testDataset).shuffle(1000).forEach(app => shuffledTestDataset.push(app));
        this.testDataset = shuffledTestDataset;

    }

    /**
     *
     * Form a batch for training the network
     *
     * @param batchSize
     * 
     */
    nextTrainBatch(batchSize) {

        this.trainBatchIndex++;
        this.checkLength('train', batchSize);

        let currentIndex = this.trainBatchIndex;
        this.trainBatchIndex = this.trainBatchIndex + batchSize;

        return this.nextBatch(
            batchSize,
            this.trainDataset,
            currentIndex
        );
    }

    /**
     *
     * Form a batch for testing the network
     *
     * @param batchSize
     * 
     */
    nextTestBatch(batchSize) {

        this.testBatchIndex++;
        this.checkLength('test', batchSize);

        let currentIndex = this.testBatchIndex;
        this.testBatchIndex = this.testBatchIndex + batchSize;

        return this.nextBatch(
            batchSize,
            this.testDataset,
            currentIndex
        );
    }

    /**
     *
     * Seperate the dataset in trainingXs and trainingYs
     *
     * @param batchSize
     * @param data test- or trainingdata
     * @param index that we not take the same data in every iteration
     * 
     */
    nextBatch(batchSize, data, index) {

        let batchXs = [];
        let batchYs = [];

        for (let i = 0; i < batchSize; i++) {

            let currentIndex = index + i;

            let x = data[currentIndex].x;
            let y = data[currentIndex].y;

            batchXs.push(x);
            batchYs.push(y);

        }

        const xs = tf.tensor(batchXs, [batchSize, 1]);
        const ys = tf.tensor(batchYs, [batchSize, 1]);

        return {
            xs,
            ys
        };
    }

    /**
     *
     * Check the BatchIndex and reset it to 0 if needed
     *
     * @param mode train/test
     * @param dataSize test- or trainingdata
     * 
     */
    checkLength(mode, batchSize) {

        if (mode === 'test') {

            if (this.testBatchIndex + batchSize > this.testDataset.length) {

                this.testBatchIndex = 0;

            }

        } else {

            if (this.trainBatchIndex + batchSize > this.trainDataset.length) {

                this.trainBatchIndex = 0;

            }

        }

    }

}
