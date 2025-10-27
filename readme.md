# CSC2114 AI ASSIGNMENT

## Project Details

**Project Title:** Diffusion-Based Large Vision Models for Contextual Event Promotion and Engagement.

**Group Code:** SW-AI-46

**Keywords**: Large Vision Models (LVM), Diffusion-Based Image Generation, Event Management and Mobilization, AI Storytelling and Narration, Immersive Virtual Event Tours

**Abstract**:

**Supervisor**: [Ggaliwango Marvin](https://www.linkedin.com/in/ggaliwango-marvin-1515b7122/)

**Model Training Notebook(Google Colab)**: [https://colab.research.google.com/drive/12XoKymFMbyozgSJ1tzb3ATDQTZ2Y3qr4?usp=sharing](https://colab.research.google.com/drive/12XoKymFMbyozgSJ1tzb3ATDQTZ2Y3qr4?usp=sharing)

**Inference Notebook(Kaggle)**: [https://www.kaggle.com/code/jimjunior/sw-ai-46-event-model-inference](https://www.kaggle.com/code/jimjunior/sw-ai-46-event-model-inference)

**Hugggingface Model**: [https://huggingface.co/jimjunior/event-diffusion-model](https://huggingface.co/jimjunior/event-diffusion-model)


**Dataset**: [https://drive.google.com/drive/folders/1zxLxcOBOkCarm8e3jXcamj1Tkk7fb271?usp=sharing](https://drive.google.com/drive/folders/1zxLxcOBOkCarm8e3jXcamj1Tkk7fb271?usp=sharing)

**Application API Docker Image**: `jimjuniorb/event-gen-model`

**Application UI Endpoint**: [https://event-gen.open.ug](https://event-gen.open.ug)


### Team Members

| Name                | Student Number | Registration Number | Student Email                                                                           | University Affiliation |
| ------------------- | -------------- | ------------------- | --------------------------------------------------------------------------------------- | ---------------------- |
| Beingana Jim Junior | 2200705243     | 22/X/5243/PS        | [beingana.jim.junior@students.mak.ac.ug](mailto:beingana.jim.junior@students.mak.ac.ug) | Makerere University    |
| Simon Peter Mujuni  | 1900708714     | 19/U/8714/EVE       | [simon.mujuni@students.mak.ac.ug](mailto:simon.mujuni@students.mak.ac.ug)               | Makerere University    |
| Boonabaana Bronia   | 2300707647     | 23/U/07647/EVE      | [boonabaana.bronia@students.mac.ac.ug](mailto:boonabaana.bronia@students.mac.ac.ug)     | Makerere University    |

## Running The Project

This project can be run using docker. As long as the host machine has access to an Nvidia GPU.

If there is no access to a GPU host machine, the group team has created a walkaround notebook that enables you to deploy the API in a Colab or Kaggle Notebook that has access to a GPU. Its in the `notebooks/deploy_api_walkaround.ipynb` file.

Once you have the API running, head over to [https://event-gen.open.ug](https://event-gen.open.ug) and paste the URL endpoint you generated to access the API in the Chatbot UI, and then start prompting the Modal.
