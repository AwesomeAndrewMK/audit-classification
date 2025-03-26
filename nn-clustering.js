const tf = require('@tensorflow/tfjs');
const { loadCSV } = require('./utils');
const { kmeans } = require('ml-kmeans');

async function run() {
    const records = await loadCSV('./data/audit_risk.csv');
    const expectedLength = Object.keys(records[0]).length - 1;

    const features = records.map(row => {
        const obj = { ...row };
        delete obj.Risk;

        const values = Object.values(obj).map(Number);
        if (values.some(isNaN) || values.length !== expectedLength) return null;

        return values;
    }).filter(Boolean);

    const trueLabels = records.map(r => Number(r.Risk)).slice(0, features.length);

    // Перетворення у Tensor
    const inputTensor = tf.tensor2d(features);

    // --- Побудова автоенкодера ---
    const inputSize = expectedLength;
    const encodingSize = 8;

    const encoder = tf.sequential();
    encoder.add(tf.layers.dense({ inputShape: [inputSize], units: encodingSize, activation: 'relu' }));
    encoder.add(tf.layers.dense({ units: inputSize, activation: 'sigmoid' }));

    encoder.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

    console.log('Навчання нейромережі...');
    await encoder.fit(inputTensor, inputTensor, {
        epochs: 50,
        batchSize: 32,
        verbose: 0,
    });
    console.log('Навчання завершено');

    // --- Витягуємо стислий вектор з першого шару ---
    const hiddenLayer = tf.model({ inputs: encoder.inputs, outputs: encoder.layers[0].output });
    const encodedTensor = hiddenLayer.predict(inputTensor);
    const encoded = await encodedTensor.array();

    // --- Кластеризація на стислих векторах ---
    const k = 2;
    const result = kmeans(encoded, k);

    // --- Оцінка точності ---
    let matchCount = 0;
    for (let i = 0; i < result.clusters.length; i++) {
        if (result.clusters[i] === trueLabels[i]) matchCount++;
    }
    const accuracy = Math.max(matchCount, encoded.length - matchCount) / encoded.length;

    console.log(`Точність кластеризації (на стислих ознаках): ${(accuracy * 100).toFixed(2)}%`);
}

run().catch(console.error);
