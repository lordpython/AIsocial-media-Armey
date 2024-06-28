import streamlit as st
import sys
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.openai import ChatOpenAI
from stream import StreamToStreamlit
import os
from dotenv import load_dotenv
import json
import logging
import pandas as pd
from datetime import datetime
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = {}

def save_config():
    with open('config.json', 'w') as f:
        json.dump(st.session_state.config, f)

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def dynamic_input(label, key, input_type='text', options=None):
    if key not in st.session_state.config:
        st.session_state.config[key] = ''

    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        if input_type == 'text':
            value = st.text_input(label, value=st.session_state.config[key], key=f"{key}_input")
        elif input_type == 'textarea':
            value = st.text_area(label, value=st.session_state.config[key], key=f"{key}_area")
        elif input_type == 'selectbox':
            value = st.selectbox(label, options, index=options.index(st.session_state.config[key]) if st.session_state.config[key] in options else 0, key=f"{key}_select")
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    with col2:
        is_active = st.checkbox("", value=bool(st.session_state.config[key]), key=f"{key}_checkbox")

    if value != st.session_state.config[key]:
        st.session_state.config[key] = value
        save_config()

    return value if is_active else ""

def initialize_llm(api, api_key, model, temp):
    try:
        if api == 'Groq':
            return ChatGroq(temperature=temp, model_name=model, groq_api_key=api_key)
        elif api == 'OpenAI':
            return ChatOpenAI(temperature=temp, openai_api_key=api_key, model_name=model)
        elif api == 'Anthropic':
            return ChatAnthropic(temperature=temp, anthropic_api_key=api_key, model_name=model)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def create_agent(role, backstory, goal, llm):
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()
    return Agent(role=role, backstory=backstory, goal=goal, allow_delegation=True, verbose=True, max_iter=3, max_rpm=3, llm=llm, tools=[search_tool, scrape_tool])

def create_task(description, expected_output, agent, context=None):
    task = Task(description=description, expected_output=expected_output, agent=agent)
    if context:
        task.context = context
    return task

def configuration_tab():
    st.header("Configuration")
    
    api_options = ['Groq', 'OpenAI', 'Anthropic']
    api = dynamic_input('Choose an API', 'api', input_type='selectbox', options=api_options)
    api_key = dynamic_input('Enter API Key', 'api_key')
    temp = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=st.session_state.config.get('temp', 0.7), step=0.1, key='temp_slider')
    st.session_state.config['temp'] = temp
    save_config()

    model_options = {
        'Groq': ['llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
        'OpenAI': ['gpt-4-turbo', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125', 'gpt-4'],
        'Anthropic': ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-haiku-20240307']
    }

    model = dynamic_input('Choose a model', 'model', input_type='selectbox', options=model_options.get(api, []))

    with st.expander("Agent Definitions", expanded=False):
        agent_1_role = dynamic_input("Agent 1 Role", "agent_1_role")
        agent_1_backstory = dynamic_input("Agent 1 Backstory", "agent_1_backstory", input_type='textarea')
        agent_1_goal = dynamic_input("Agent 1 Goal", "agent_1_goal", input_type='textarea')
        
        agent_2_role = dynamic_input("Agent 2 Role", "agent_2_role")
        agent_2_backstory = dynamic_input("Agent 2 Backstory", "agent_2_backstory", input_type='textarea')
        agent_2_goal = dynamic_input("Agent 2 Goal", "agent_2_goal", input_type='textarea')
        
        agent_3_role = dynamic_input("Agent 3 Role", "agent_3_role")
        agent_3_backstory = dynamic_input("Agent 3 Backstory", "agent_3_backstory", input_type='textarea')
        agent_3_goal = dynamic_input("Agent 3 Goal", "agent_3_goal", input_type='textarea')

    return api, api_key, temp, model, agent_1_role, agent_1_backstory, agent_1_goal, agent_2_role, agent_2_backstory, agent_2_goal, agent_3_role, agent_3_backstory, agent_3_goal

def execution_tab(api, api_key, temp, model, agent_1_role, agent_1_backstory, agent_1_goal, agent_2_role, agent_2_backstory, agent_2_goal, agent_3_role, agent_3_backstory, agent_3_goal):
    st.header("Execution")
    
    var_1 = dynamic_input("Variable 1:", "var_1")
    var_2 = dynamic_input("Variable 2:", "var_2")
    var_3 = dynamic_input("Variable 3:", "var_3")

    if st.button("Start", disabled=not (api_key and any([var_1, var_2, var_3]))):
        with st.spinner("Generating..."):
            try:
                llm = initialize_llm(api, api_key, model, temp)
                if not llm:
                    st.warning("Failed to initialize LLM. Please check your API key and selected model.")
                    return

                agents = []
                tasks = []

                for i, (role, backstory, goal) in enumerate([(agent_1_role, agent_1_backstory, agent_1_goal),
                                                             (agent_2_role, agent_2_backstory, agent_2_goal),
                                                             (agent_3_role, agent_3_backstory, agent_3_goal)], 1):
                    if role and backstory and goal:
                        agent = create_agent(role, backstory, goal, llm)
                        agents.append(agent)
                        task = create_task(
                            description=f"Task for Agent {i}:\n---\nVARIABLE 1: {var_1}\nVARIABLE 2: {var_2}\nVARIABLE 3: {var_3}",
                            expected_output="A detailed output based on the agent's role and given variables.",
                            agent=agent,
                            context=tasks if tasks else None
                        )
                        tasks.append(task)

                if not agents or not tasks:
                    st.warning("No valid agents or tasks created. Please check your agent configurations.")
                    return

                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    verbose=2,
                    process=Process.hierarchical,
                    manager_llm=llm
                )

                output_expander = st.expander("Output", expanded=True)
                original_stdout = sys.stdout
                sys.stdout = StreamToStreamlit(output_expander)

                result = ""
                result_container = output_expander.empty()
                for delta in crew.kickoff():
                    result += delta
                    result_container.markdown(result)
                
                # Save results to a CSV file
                results_df = pd.DataFrame({
                    'Timestamp': [datetime.now()],
                    'API': [api],
                    'Model': [model],
                    'Temperature': [temp],
                    'Variable 1': [var_1],
                    'Variable 2': [var_2],
                    'Variable 3': [var_3],
                    'Result': [result]
                })
                results_df.to_csv('results.csv', mode='a', header=not os.path.exists('results.csv'), index=False)
                
                logging.info("CrewAI process completed successfully")
                st.success("Process completed successfully!")

            except Exception as e:
                logging.error(f"An error occurred during execution: {str(e)}")
                st.error(f"An error occurred during execution: {str(e)}")
            finally:
                sys.stdout = original_stdout

def results_tab():
    st.header("Results")
    
    try:
        results_df = pd.read_csv('results.csv')
        st.write("Here are the latest results from your CrewAI executions:")
        st.dataframe(results_df)
        
        if st.button("Download Results CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Click here to download",
                data=csv,
                file_name="crewai_results.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.info("No results found. Run an execution to generate results.")
    
    st.subheader("Log Output")
    try:
        with open("app.log", "r") as log_file:
            st.code(log_file.read())
    except FileNotFoundError:
        st.info("No log file found. Run an execution to generate a log.")

def main():
    st.set_page_config(page_title="Yousef AI Army", page_icon="🤖", layout="wide")
    
    st.title('Yousef AI Army')
    st.markdown("This program is an army of AI robots at your disposal.")

    col1, col2 = st.columns([5, 1])
    with col2:
        st.image('logo.png')

    tab1, tab2, tab3 = st.tabs(["Configuration", "Execution", "Results"])

    with tab1:
        config_params = configuration_tab()

    with tab2:
        execution_tab(*config_params)

    with tab3:
        results_tab()

if __name__ == "__main__":
    main()
