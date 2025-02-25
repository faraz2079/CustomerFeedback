import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { SharedArray } from 'k6/data';

// Read and parse the JSON input file
const inputData = new SharedArray('input data', function () {
    return JSON.parse(open('/Users/happy/Documents/Project Thesis/project/CustomerFeedback/file_processing/inputFile1.json'));
});

export const options = {
    scenarios: {
        bulk_requests: {
            executor: 'shared-iterations',
            iterations: inputData.length,
            vus: 1,
        },
    },
    teardownTimeout: '60s',
};


export default function () {
    const record = inputData[__ITER];
    const payload = JSON.stringify(record);
    const url = 'http://172.22.174.240:8501/api/v1/feedback';
    //const url = 'http://localhost:8501/api/v1/feedback';
    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const res = http.post(url, payload, params);

    console.log(`Customer Review: ${payload}`);
    console.log(`**** Model predicts: Customer is ${res.body} ****`);
}

export function teardown() {
    const url = 'http://172.17.0.1:32501/uploadInputFile';
    //const url = 'http://localhost:8000/uploadInputFile';
    const res = http.get(url);
    if (res.status === 200) {
        console.log('File successfully uploaded to S3.');
    } else {
        console.error(`Failed to upload file. Status: ${res.status}, Response: ${res.body}`);
    }
}
