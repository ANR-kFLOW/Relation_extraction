#  Copyright 2024. EURECOM
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# === Import ==================================================================

import logging
import os
import re
import time
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import yaml
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from LocalTemplate import LocalTemplate
print('done')

# === Init ====================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"  # Specify the GPU id
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huggingface/"

available_llms = {
    "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",

}


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        formatted_prompts = "\n".join(prompts)
        print(f"********** Prompt **************\n{formatted_prompts}\n********** End Prompt **************")


def run(llm_model, verbose=False, output_path='/out', n_examples=0):
    """Processing entry point: load the ontology, the prompting template, then call the LLM and save the results.


    :param llm_model:
    :param output_path:
    :param id:
    :param n_examples:
    :return:
    """

    template_path = f'prompt_template.yml'
    PROMPT_TEMPLATE = LocalTemplate.load(template_path)
    logging.debug("LOAD:PROMPT_TEMPLATE:template=%s", PROMPT_TEMPLATE)
    df_examples = pd.read_csv('clean.csv')
    df = pd.DataFrame()
    while 1:
        random_examples_explicit = df_examples.sample(n=n_examples)
        #    random_examples_implit = df_implit.sample(n=2)
        #    combined_df = pd.concat([random_examples_explicit, random_examples_implit])

        # random_selection = df.sample(n=5)
        # print('len df', len(df))

        # Storing selected sentences in a list
        combined_text = ""
        for i, text in enumerate(random_examples_explicit['text'], start=1):
            combined_text += f"{i}. {text}\n"

        examples = combined_text

        logging.debug("LOAD:CQS:n_examples=%s:examples=%s", n_examples, examples)

        ont_input = {

            'examples': examples
        }
        input_dict = {}
        for x in PROMPT_TEMPLATE.input:
            input_dict[x] = ont_input[x]

        tokenizer = AutoTokenizer.from_pretrained(available_llms[llm_model])

        model = AutoModelForCausalLM.from_pretrained(
            available_llms[llm_model],
            device_map='sequential',
            load_in_8bit=True,
            use_safetensors=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        output_parser = StrOutputParser()
        prompt = PromptTemplate(
            input_variables=["prompt"] + PROMPT_TEMPLATE.input, template=PROMPT_TEMPLATE.get()
        )
        chain = prompt | llm | output_parser

        # Call LLM
        logging.info("PROMPT:CALL_LLM:%s:engine=%s", 'START', llm)
        config = {"callbacks": [CustomHandler()]} if verbose else {}
        res = chain(config=config)
        logging.info("PROMPT:CALL_LLM:%s:res=%s", 'DONE', res)

        # Parse output to get the generated sentences
        sentences = re.split(r'\n\s?\d+\.\s*', res)[1:]
        new_sentences_df = pd.DataFrame({'Sentences': sentences})
        df = pd.concat([df, new_sentences_df], ignore_index=True)

        df.to_csv('prevention_new.csv', index=False)

        return True, res


# === Main ====================================================================

if __name__ == '__main__':
    # Get tasks from prompt templates

    # Define argument parser
    parser = ArgumentParser(
        prog='DA for Common Sense',
        description='DA for Common sense knowledge for event relations: enable and prevent'
    )

    parser.add_argument(
        '-o',
        '--output',
        help='Output folder',
        default='./out/'
    )

    parser.add_argument(
        '--llm',
        help='LLM to use',
        default='zephyr',
        choices=available_llms
    )

    parser.add_argument(
        '-x',
        '--n_examples',
        help='Number of example competency questions to provide in input',
        type=int,
        default=2
    )

    parser.add_argument(
        '--verbose',
        help='Print the full prompt',
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "--log",
        type=int,
        choices=[10, 20, 30, 40, 50],
        action="store",
        default=20,
        help="Verbosity (default: INFO) : DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50",
    )

    # Instanciate argument parser
    args = parser.parse_args()
    _llm = args.llm



    # Call the processing function
    logging.info("INIT")
    run_response = run(

        _llm,

        args.verbose,
        args.output,
       
        args.n_examples
    )

    logging.info("DONE")
