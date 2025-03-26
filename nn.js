const tf = require('@tensorflow/tfjs');
const { loadCSV } = require('./utils');

function normalize(tensor) {
    const { mean, variance } = tf.moments(tensor, 0);
    return tensor.sub(mean).div(variance.sqrt());
}

(async () => {
    const rawData = await loadCSV('./data/audit_risk.csv');

    const allData = rawData.map(row => {
        const values = Object.values(row).map(Number);
        return values.map(v => (isNaN(v) ? 0 : v));
    });

    const splitIndex = Math.floor(allData.length * 0.8);
    const trainData = allData.slice(0, splitIndex);
    const testData = allData.slice(splitIndex);

    const trainX = trainData.map(r => r.slice(0, -1));
    const trainY = trainData.map(r => r.at(-1));

    const testX = testData.map(r => r.slice(0, -1));
    const testY = testData.map(r => r.at(-1));

    const xs = normalize(tf.tensor2d(trainX));
    const ys = tf.tensor2d(trainY, [trainY.length, 1]);

    const testXs = normalize(tf.tensor2d(testX));
    const testYs = tf.tensor2d(testY, [testY.length, 1]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [trainX[0].length] }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    console.log('Навчання моделі...');
    await model.fit(xs, ys, {
        epochs: 30,
        batchSize: 32,
        shuffle: true,
        verbose: 1,
    });

    const [_, acc] = await model.evaluate(testXs, testYs);
    console.log(`Точність моделі (nn): ${(acc.dataSync()[0] * 100).toFixed(2)}%`);
})();
