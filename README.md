first deploy the application using the deployment and service file in inference_pipeline, then we can run the script and it applies deployment files automatically for testing purposes. 


for running the script:() 
./run_experiment.sh | tee -a experiment.log


sudo pkill -f wrk

ps aux | grep wrk

in this command you have to give the path to the wrk2 directory: 
/home/ubuntu/DeathStarBench/wrk2/wrk -t96 -c10000 -d300s -R 10000 -s ./mixed-workload.lua http://192.168.1.243:30915 > wrk_R10000_5min.log &



# CustomerFeedback
This project work performs sentiment analysis on customer feedback for the product they have purchased.
Based on the review and stars that the customer has provided, the machine determines whether the customer is happy,
satisfied, angry, disappointed or neutral.

This project work uses K6 for simulating bulk requests, ML model (trainer), inference and an intermittent java microservice. 
