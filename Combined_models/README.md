# Use the Docker image

Create and populate the `pretrained_models` folder. TBW How?

Build the docker image using

    docker build -t kflow/rel_extraction .

then run the image with

    docker run -p 5004:5004 -v $(pwd)/pretrained_models:/pretrained_models -v $(pwd)/out:/out --name kflow_rel_extraction kflow/rel_extraction 



When running the link that is given in the terminal change the local port number to the number that you exported from your local machine. 
for the previous example it would be http://127.0.0.1:5004/swagger/

(Be sure to add the /swagger to the end of the link)






Pretrained Models storage and directory pathing: 
The heirarchy for storing the pretrained models should look like
pretrained_models:
  st0:
    roberta_st0:
      pretrained_model.pt
  st1:
    roberta_st1:
      pretrained_model_folder
  st2:
    roberta_st2:
      pretrained_model_folder
  st3:
    LLM_st3:
      nothing so far
    rebel_st3:
      pretrained_model.pth

St0: filters out the entries that have no relationship
St1: classifies the type of relationship in a given entry
St2: gives the subject and the object of a given relationship for each entry
St3: these models output a combination of St1 and St2

In the options St3 models can be used to do either st1 or st2 tasks, however when the model is called both tasks are done regardless if the pipeline requests for only one.
So far the LLM_st3 options do not require anything from the folder but it is still kept because a future LLM might need something stored in there.

All of the place holder names would have to be replaced with a counterpart that is compatible with the script that runs the model. 
When putting in files into st0 or rebel_st3 be mindful of keeping the file as the same file type as shown in the example. 

When putting in multiple pretrained model files in the same directory make sure that each name is unique so that it can be clearly distinguished when the model appears as an option in the web application.



When importing this github be sure to create the empty folders: saved_app_outs and combined_outs
