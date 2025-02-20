# NLPCC 2025 Shared Task8

## Task Overview
Emotional Support Conversation (ESC) systems are designed to alleviate users' emotional distress and assist them in managing various challenges. However, traditional ESC systems often generate generic responses that fail to address users' unique, personalized needs. To address this issue, a new task called Personalized Emotional Support Conversation (PESC) is introduced. PESC systems consider the personal traits of users to generate emotionally supportive responses tailored to various individuals. The effectiveness of these systems is assessed using both automatic metrics and LLM-based metrics, ensuring a comprehensive evaluation.


## Data Description
We provide a dataset containing 1,500 dialogue instances, divided into training, validation, and test sets in an 8:1:1 ratio. All data is available in this repository. The required file structure is as follows:
```
data
├── train.json                     # training dataset
├── val.json                       # validation dataset
└── test.json                      # test dataset
```
Each dialogue instance contains 20 conversational turns along with the Seeker’s personal profile, structured as shown below:
```
"whole_dialog": [
    "seeker:I feel really isolated and anxious because I can't see my friends due to COVID-19, and it makes me missthem even more.",
    "supporter:It's completely understandable to feel anxious and isolated during such challenging times. Manypeople are experiencing similar feelings right now. Setting up a virtual hangout with friends or family canreally help bridge that gap! How about we brainstorm some specific activities you could do during those hangouts",
    "seeker:That sounds great; I think a virtual game night could really lift my spirits and help me feel connectedwith them. I would rate that a 5 because it sounds like a fun way to spend time together!",
    "supporter:That sounds like a fantastic idea! Blending writing sessions with fun activities like game nightswill definitely enhance your connections. Let\u2019s dive into those ideas and create some engaging planstogether!",
    ...
    ...
],
"character": {
    "Name": "Samantha Carter",
    "Sex": "Female",
    "Nickname": "Sam",
    "Age": "29",
    "Occupation": "Customer Service Representative",
    "Hobby": "Writing",
    "Appearance": "Tall, brown hair, wears casual clothes",
    "Education Background": "B.A. in Communications",
    "emotion_type": "Anxious",
    "problem_type": "Job Crisis",
    "situation": "I can not hang out with my friends due to COVID-19. I miss them a lot but they are high risk."
}
```
The test dataset does not contain the supporter's responses.


## Evaluation
For evaluation, the automated evaluation metrics we use include **METEOR (ME)**, **BLEU-4 (B-4)**, **ROUGE-L (R-L)**, **VectorExtrema (Ext)**, and **Distinct2/3 (D-2/3)**. We provide a script, `eval.py`, to compute these metrics and generate a total score for the validation dataset. The total score is calculated as follows:
$$\text{score} = 0.15 * \text{ME} + 0.25 * \text{B-4} + 0.25 * \text{R-L} + 0.15 * \text{Ext} + 0.1 * \text{D-2} + 0.1 * \text{D-3}$$


## Submission
For submission, the following materials should be packaged as one zip file and sent to <220110515@stu.hit.edu.cn>:
- **Dilogue results**: Your model's output should be filled into the corresponding blank Supporter field in the `test.json` file. The output file should be renamed as `result.json`.
- **Code**: All code files related to data processing, model training, model inference, and other related processes.
- **Description**: Necessary explanatory documents (describing the code structure, how to run the code to get your results, etc.).


## Contact
If you have any questions about this task, please email to (<220110515@stu.hit.edu.cn> (C.C. <bingbing.wang@stu.hit.edu.cn>, <22b951011@stu.hit.edu.cn>).

