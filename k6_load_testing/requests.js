import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { SharedArray } from 'k6/data';

// Read and parse the JSON input file
const inputData = new SharedArray('input data', function () {
    return JSON.parse(open('/home/bhanu/CustomerFeedback/file_processing/inputFile1.json'));
});

export const options = {
    scenarios: {
        bulk_requests: {
            executor: 'constant-arrival-rate',
            rate: 150, // requests per second
            timeUnit: '1s',
            duration: '10m',
            preAllocatedVUs: 150,
            maxVUs: 300,
        },
    },
    thresholds: {
        http_req_duration: ['p(90)<6000'], // 90% of requests should complete below 6s
        http_req_failed: ['rate<0.01'], // less than 1% requests should fail
    },
};


export default function () {
    const record = inputData[__ITER % inputData.length];
    const payload = JSON.stringify(record);

    const url = 'http://localhost:8501/api/v1/feedback';
    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const res = http.post(url, payload, params);

    console.log(`Customer Review: ${payload}`);
    console.log(`**** Review Analysis: ${res.body} ****`);
    console.log(`Response status: ${res.status}`);
    console.log(`Response time: ${res.timings.duration}ms`);

    // Validate response
    check(res, {
    'status is 200': (r) => {
        const statusCheck = r.status === 200;
        if (!statusCheck) {
            console.error(`Unexpected status: ${r.status}, Response: ${r.body}`);
        }
        return statusCheck;
    },
    'response time < 5000ms': (r) => {
        const timingCheck = r.timings.duration < 5000;
        if (!timingCheck) {
            console.log(`Slow response: ${r.timings.duration}ms, Response: ${r.body}`);
        }
        return timingCheck;
    },
});

    sleep(1);
}

export function teardown() {
    const url = 'http://localhost:32501/uploadInputFile';
    const res = http.get(url);
    if (res.status === 200) {
        console.log('File successfully uploaded to S3.');
    } else {
        console.error(`Failed to upload file. Status: ${res.status}, Response: ${res.body}`);
    }
}
