First deploy the application using the deployment and service file in inference_pipeline, then we can run the script and it applies deployment files automatically for testing purposes. 

- The script runs for 63 minutes and it runs 3 load cycles (low - mid - high):
- every load cycle contains 3 cycles with 2 minutes pause in between and also 2 minutes pause between the load cycles.
- it saves the result in the files in the directory that the script exists

**for running the script:** 

foreground : `./run_experiment.sh | tee -a experiment.log`

background: `nohup ./run_experiment.sh > experiment.out 2>&1 &` 

track the background process: `tail -f experiment.out`

track if the process is running in the background using PID: `ps -p 200502 -f`

**Commands for killing and tracking the process of script:**

`sudo pkill -f wrk`

`ps aux | grep wrk`

in this command you have to give the path to the wrk2 directory: => give the url of the customerfeedback service and url of the VM. 
`/home/ubuntu/DeathStarBench/wrk2/wrk -t96 -c10000 -d300s -R 10000 -s ./mixed-workload.lua http://192.168.1.243:30915 > wrk_R10000_5min.log &`


# CustomerFeedback
This project work performs sentiment analysis on customer feedback for the product they have purchased.
Based on the review and stars that the customer has provided, the machine determines whether the customer is happy,
satisfied, angry, disappointed or neutral.

This project work uses K6 for simulating bulk requests, ML model (trainer), inference and an intermittent java microservice. 
