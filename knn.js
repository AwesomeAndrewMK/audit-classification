function prepareData(rawData) {
    const features = [];
    const labels = [];

    rawData.forEach(row => {
        const rowValues = Object.values(row).map(v => parseFloat(v));
        const input = rowValues.slice(0, -1); // щоб не використовувати Risk
        const output = rowValues[rowValues.length - 1];
        if (!isNaN(output)) {
            features.push(input);
            labels.push(output);
        }
    });

    return { features, labels };
}

function classify(knn, inputRow) {
    return knn.predict([inputRow])[0];
}

module.exports = { prepareData, classify };
