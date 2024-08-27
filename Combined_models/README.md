## Sub-task list

The pipeline adresses the following sub-tasks:
- `st0` filters out the entries that have no relationship
- `st1` classifies the type of relationship in a given entry among `cause`, `enable`, `intend`, or `prevent`
- `st2` identify the subject and the object of a given relationship for each entry
- `st3` these models output a combination of `st1` and `st2`

In the options, `st3` models can be used to do either st1 or st2 tasks.
However, when the model is called both tasks are done regardless if the pipeline requests for only one.


## Import the pretrained models

Create and populate the `pretrained_models` folder.
The hierarchy for storing the pretrained models should look like
```
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
```

The LLM_st3 options does not require anything for the moment. but it is still kept because a future LLM might need something stored in there.

When adding new files into `st0` or `rebel_st3` be mindful of keeping the file as the same file type as shown in the example. 



TBW where to find these files

## Use the Docker image

Build the docker image using

    docker build -t kflow/rel_extraction .
    
    # in production on linux
    docker build --build-arg="ENV_FILE=environment_prod.yml" -t kflow/rel_extraction .

then run the image with

    docker run -d -p 5002:5004 -v $(pwd)/pretrained_models:/pretrained_models -v $(pwd)/out:/out --name kflow_rel_extraction kflow/rel_extraction

When running the link that is given in the terminal change the local port number to the number that you exported from your local machine. 
for the previous example it would be http://127.0.0.1:5004/swagger/
> (Be sure to add the /swagger to the end of the link)



## Running the pipeline manually

There are two options for running the pipeline without the use of the web application: passing the arguements to call_pipeline.py, and creating a config file which is passed to call_pipeline.py

Arguements available:

  Input the path to the file you want to use for inferences
  
    --test_file
    
  Input the path to the pretrained model you are using for st1. Be sure to include the path in the pretrained models folder
  
    --st1_mod
    
  Input the path to the pretrained model you are using for st2.  
    
    --st2_mod

  If there is a config file available input the path and that will be taken as a priority over the other inputs. 
    
    --config_path
  
  Type out your sentences in quote marks and be sure to seperate them with periods. This will be used instead of the test file
    
    --text_from_user
  
  Type in true if you do not want the pipeline to perform st1
    
    --skip_st1
  
  Type in true if you do not want the pipeline to perform st2
    
    --skip_st2
  
    

    





