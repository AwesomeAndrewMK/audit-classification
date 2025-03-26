const { kmeans } = require('ml-kmeans');
const { loadCSV } = require('./utils');

async function main() {
    const records = await loadCSV('./data/audit_risk.csv');
    const expectedLength = Object.keys(records[0]).length - 1; // мінус Risk

    const features = records.map(row => {
        const obj = { ...row };
        delete obj.Risk;

        const values = Object.values(obj).map(Number);
        if (values.some(isNaN) || values.length !== expectedLength) return null;

        return values;
    }).filter(Boolean);

    // Зберігаємо справжні значення Risk
    const trueLabels = records.map(r => Number(r.Risk));

    // Кількість кластерів
    const k = 2;
    const result = kmeans(features, k);

    // Оцінка точності
    let matchCount = 0;
    for (let i = 0; i < result.clusters.length; i++) {
        if (result.clusters[i] === trueLabels[i]) matchCount++;
    }

    const accuracy = Math.max(matchCount, records.length - matchCount) / records.length;
    console.log(`Точність кластеризації (порівняно з Risk): ${(accuracy * 100).toFixed(2)}%`);
}

main()
    .catch(console.error);
