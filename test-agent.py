import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import tempfile
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import matplotlib.pyplot as plt
import requests
from PIL import Image
from termcolor import colored

import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.llava_agent import LLaVAAgent, llava_call

LLAVA_MODE = "remote"  # Either "local" or "remote"
assert LLAVA_MODE in ["local", "remote"]

# Run this code block only if you want to run LlaVA locally
llava_config_list = [
    {
        "model": "llava-v1.5-13b",
        "api_key": "None",
        "base_url": "http://0.0.0.0:10000",
    }
]
    
rst = llava_call(
    "Describe this AutoGen framework <img https://raw.githubusercontent.com/microsoft/autogen/main/website/static/img/autogen_agentchat.png> with bullet points.",
    llm_config={"config_list": llava_config_list, "temperature": 0},
)

print(rst)
    
class FigureCreator(AssistantAgent):
    def __init__(self, n_iters=2, **kwargs):
        """
        Initializes a FigureCreator instance.

        This agent facilitates the creation of visualizations through a collaborative effort among its child agents: commander, coder, and critics.

        Parameters:
            - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 2.
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.register_reply([Agent, None], reply_func=FigureCreator._reply_user, position=0)
        self._n_iters = n_iters

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)  # noqa: F821
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        user_question = messages[-1]["content"]

        ### Define the agents
        commander = AssistantAgent(
            name="Commander",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="Help me run the code, and tell other agents it is in the <img result.jpg> file location.",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "last_n_messages": 3,
                "work_dir": ".",
                "use_docker": False,
            },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
            llm_config=self.llm_config,
        )

        critics = LLaVAAgent(
            name="Critics",
            system_message="""Criticize the input figure. How to replot the figure so it will be better? Find bugs and issues for the figure.
            Pay attention to the color, format, and presentation. Keep in mind of the reader-friendliness.
            If you think the figures is good enough, then simply say NO_ISSUES""",
            llm_config={"config_list": llava_config_list},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            #     use_docker=False,
        )

        coder = AssistantAgent(
            name="Coder",
            llm_config=self.llm_config,
        )

        coder.update_system_message(
            coder.system_message
            + "ALWAYS save the figure in `result.jpg` file. Tell other agents it is in the <img result.jpg> file location."
        )

        # Data flow begins
        commander.initiate_chat(coder, message=user_question)
        img = Image.open("result.jpg")
        plt.imshow(img)
        plt.axis("off")  # Hide the axes
        plt.show()

        for i in range(self._n_iters):
            commander.send(message="Improve <img result.jpg>", recipient=critics, request_reply=True)

            feedback = commander._oai_messages[critics][-1]["content"]
            if feedback.find("NO_ISSUES") >= 0:
                break
            commander.send(
                message="Here is the feedback to your figure. Please improve! Save the result to `result.jpg`\n"
                + feedback,
                recipient=coder,
                request_reply=True,
            )
            img = Image.open("result.jpg")
            plt.imshow(img)
            plt.axis("off")  # Hide the axes
            plt.show()

        return True, "result.jpg"
    

    
env_var = [
    {
        'model': 'gpt-4',
        'api_key': os.environ.get("OAI_API_KEY")
    },
    {
        'model': 'gpt-3.5-turbo',
        'api_key': os.environ.get("OAI_API_KEY")
    }
]

with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp:
    env_var = json.dumps(env_var)
    temp.write(env_var)
    temp.flush()

    # Setting configurations for autogen
    config_list_gpt4 = autogen.config_list_from_json(
        env_or_file=temp.name,
        filter_dict={
            "model": {
                "gpt-4",
                "gpt-3.5-turbo",
            }
        }
    )
    
# config_list_gpt4 = autogen.config_list_from_json(
#     env_or_file=env_var,
#     filter_dict={
#         "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
#     },
# )

gpt4_llm_config = {"config_list": config_list_gpt4, "cache_seed": 42}

# config_list_gpt35 = autogen.config_list_from_json(
#     "OAI_CONFIG_LIST",
#     filter_dict={
#         "model": ["gpt-35-turbo", "gpt-3.5-turbo"],
#     },
# )

# gpt35_llm_config = {"config_list": config_list_gpt35, "cache_seed": 42}


creator = FigureCreator(name="Figure Creator~", llm_config=gpt4_llm_config)

user_proxy = autogen.UserProxyAgent(
    name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config={"use_docker": False}
)  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.

user_proxy.initiate_chat(
    creator,
    message="""
Plot a figure by using the data from:
https://raw.githubusercontent.com/vega/vega/main/docs/data/seattle-weather.csv

I want to show both temperature high and low.
""",
)