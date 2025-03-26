const { loadCSV } = require('./utils');
const { prepareData, classify } = require('./knn');
const KNN = require('ml-knn');

(async () => {
    const data = await loadCSV('./data/audit_risk.csv');
    const { features, labels } = prepareData(data);

    const splitIndex = Math.floor(features.length * 0.8);

    const trainX = features.slice(0, splitIndex);
    const trainY = labels.slice(0, splitIndex);

    const testX = features.slice(splitIndex);
    const testY = labels.slice(splitIndex);

    const knn = new KNN(trainX, trainY, { k: 3 });

    let correct = 0;

    testX.forEach((x, i) => {
        const predicted = classify(knn, x);
        const actual = testY[i];
        if (predicted === actual) correct++;

        console.log(`Об'єкт ${i + 1}: Передбачено = ${predicted}, Реально = ${actual}`);
    });

    const accuracy = (correct / testX.length) * 100;
    console.log(`Точність моделі (k-NN): ${accuracy.toFixed(2)}%`);
})();
