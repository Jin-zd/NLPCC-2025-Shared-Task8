# NLPCC 2025 Shared Task8

## Task Overview
Emotional Support Conversation (ESC) systems are designed to alleviate users' emotional distress and assist them in managing various challenges. However, traditional ESC systems often generate generic responses that fail to address users' unique, personalized needs. To address this issue, a new task called Personalized Emotional Support Conversation (PESC) is introduced. PESC systems consider the personal traits of users to generate emotionally supportive responses tailored to various individuals. The effectiveness of these systems is assessed using both automatic metrics and LLM-based metrics, ensuring a comprehensive evaluation.


## Important Dates
| Date | News |
| ---  | ---  |
| 2025/02/17 | Announcement of shared tasks and call for participation |
| 2025/02/17 | Registration open |
| 2025/02/28 | Release of detailed task guidelines & training data |
| 2025/03/25 | Registration deadline |
| 2025/04/11 | Release of test data |
| 2025/04/20 | Participants’ results submission deadline |
| 2025/04/30 | Evaluation results release and call for system reports and conference paper |
| 2025/05/22 | Conference paper submission deadline (only for shared tasks) |
| 2025/06/12 | Conference paper accept/reject notification |
| 2025/06/25 | Camera-ready paper submission deadline |


## Data Description
We provide a dataset containing 1,500 dialogue instances, divided into training, validation, and test sets in an 8:1:1 ratio. All data is available in this repository. The required file structure is as follows:
```
data
├── train.json                     # training dataset
├── val.json                       # validation dataset
└── test.json                      # test dataset (will be released on 2025/04/11)
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

**In order to ensure the proper handling and protection of personal information, the data cannot be used for commercial purposes outside of the competition. For any other academic purposes, please contact the organizers for further discussion.**


## Evaluation
For evaluation, the automated evaluation metrics we use include **METEOR (ME)**, **BLEU-4 (B-4)**, **ROUGE-L(R-L)**, **VectorExtrema (Ext)**,and **Distinct2/3 (D-2/3)**. GPT4 will evaluate the relevance, fluency, informativeness, and logical coherence of your responses to get a score(**G-Score**). We provide a script, `eval.py`, to compute these metrics and generate a total score for the validation dataset. The total score is calculated as follows:

$$\text{score} = 0.2 * \text{ME} + 0.2 * \text{B-4} + 0.2 * \text{R-L} + 0.1 * \text{Ext} + 0.1 * \text{D-2} + 0.1 * \text{D-3} +  0.1 * \text{G-Score}$$


## Submission
For submission, the following materials should be packaged as one zip file and submit to <https://f.wps.cn/g/RhyrdzJj>:
- **Dilogue results**: Your model's output should be filled into the corresponding blank Supporter field in the `test.json` file. The output file should be renamed as `result.json`. 
- **Code**: All code files related to data processing, model training, model inference, and other related processes.
- **Description**: Necessary explanatory documents (describing the code structure, how to run the code to get your results, etc.).

Here is an example of dilogue results:

The `test.json` will be like:
```
{
    "1351": {
        "whole_dialog": [
            "seeker:I feel utterly abandoned and trapped in my loneliness after my boyfriend left me without any reason.",
            "supporter:"
            "seeker:I'm not sure where to start, but every day feels heavier without him, and I just don't know how to cope with this loneliness. It feels like I'm stuck in this dark place with no way out.",
            "supporter:"
            ...
        ],
        "character": {
            ...
        }
    },
    ...
    ...
}
```
And then the `result.json` should be like:
```
{
    "1351": {
        "whole_dialog": [
            "seeker:I feel utterly abandoned and trapped in my loneliness after my boyfriend left me without any reason.",
            "supporter:It sounds incredibly difficult to feel so abandoned and trapped in loneliness right now. Those feelings of isolation are completely valid. It\u2019s important to explore what these emotions mean for you and how we can begin to work through them together. Remember, you're not alone in this; let's talk about it.",
            "seeker:I'm not sure where to start, but every day feels heavier without him, and I just don't know how to cope with this loneliness. It feels like I'm stuck in this dark place with no way out.",
            "supporter:It sounds incredibly heavy to feel that way every day, and it's completely understandable to be struggling with such deep emotions. You're not alone in this, and I\u2019m here to help you explore those feelings more deeply. Let\u2019s talk about how we can start navigating through this pain together and work towards finding those small glimmers of hope.",
            ...
        ],
        "character": {
            ...
        }
    },
    ...
    ...
}
```

The submission dates are as follows: 
- April 12, 23:59
- April 14, 23:59
- April 16, 23:59
- April 18, 23:59
- April 20, 23:59, **Final submission**

For each of these dates, we will select the last submission of the day as the final result for that day, and among multiple submissions across the days, we will choose the best result.
We will publish the rankings on GitHub the day after each submission, based on each team's best submission so far, along with the results of their last submission from the previous day.


## Contact
If you have any questions about this task, please email to (<220110515@stu.hit.edu.cn> (C.C. <bingbing.wang@stu.hit.edu.cn>, <22b951011@stu.hit.edu.cn>) or submit an issue on github.

