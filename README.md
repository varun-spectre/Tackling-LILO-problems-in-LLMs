# LILO: Designing Prompting Technique for Handling Long Input and Output in Biomedical

## Task

Widespread adoption of Electronic Health Records (EHRs), Patient summary reports and research literature in a clinical domain has fueled the development of using Natural Language Processing (NLP) to build prediction models for various clinical outcomes. Recently, Large Language Models (LLMs) such as BART, T5, GPT-J, GPT-2, FLAN, T0, and GPT-3 have outperformed many benchmarks and performed well in many generation tasks. However, leveraging these models in the biomedical domain is still a challenging task since EHRs or clinical data have long inputs, long outputs or both. A recent study [Pre-train, Prompt, and Predict](https://arxiv.org/pdf/2107.13586.pdf) (Page 22) shows that predicting large output label space is a challenge in many Information Extraction (IE) tasks, especially, in the biomedical domain. Here, our task is to build effective prompting techniques that can handle large inputs as well as large output space in different biomedical tasks.

## Example

This example is taken from n2c2 Challenge Track 1 - [2014 - De-identification dataset](https://www.sciencedirect.com/science/article/pii/S1532046415001173?via%3Dihub). De-identification is the process used to prevent someone's personal identity from being revealed. So the task in this dataset is to first predict personal information such as patient name, doctor name, date, hospital name, etc. from the given text and remove it. Here, you can see that input is very long. The output is in entity <entity class> format which is also long and covers large output label space. See the below example.

#### Input

```
Record date: 2091-07-28 Intern Transfer Note Mitchell, Katie MR# 6146161 Date of Admission: July 24, 2091 Date of Transfer: Jul 28, 2091 Pulm: Geoffrey Lucas PCP: Evelyn Conway (Raymond) Cards: Janssen ID/CC: SOB HPI: Ms. Mitchell is a 75 yo woman w/ a PMH notable for COPD w/ FEV1/FVC=63%, HOCM s/p MVR and septal myomectomy in 2085 who presents with increasing SOB and DOE for the past 4 days. Of note, her and her son note increasing DOE for the past 4 months and she was scheduled for outpatient TTE today. She also reports DOE and productive cough with green- yellow sputum that has been blood tinged ever since her EGD on 7/10/91. She also notes recent URI symptoms especially runny nose as well as diarrhea the day PTA with 6 loose BMs. On the day prior to admission, she had drenching sweats but didn't take her temperature. She has also had decreased po intake (though improved w/ treatment for her trigeminal neuralgia) but denies F/C/CP/abd pain/dysuria. Her symptoms acutely worsened today such that she was in acute respiratory distress and EMS was called..\t \tEMS was called and the patient was noted to initially be speaking in full sentences but became more acutely SOB in the ambulance. In the ED, she was afebrile but tachycardic to 120 and hypertensive at 180/80. She initially required bipap with O2 sats of 93%. She was given prednisone, Levaquin and nebulizers with mild improvement in her symptoms. Of note, she was also noted to have mild ST changes on her EKG while she was tachycardic to the 120s. She received ASA, Lopressor and Ativan and cardiac enzymes showed an elevated CK and CK-MB but negative troponin at 0.03. Her EKG improved with rate control w/ IV Lopressor. She was also given 30mg IV Lasix given some mild LE edema to which she put out 200cc urine. She is now admitted to the floor for further management of her SOB. ROS: No F/C, + NS. Improving appetite. No CP/Palpitations/Orthopnea/PND/Claudication, + LE edema for the past 1-2 months. No N/V/D/hematochezia/melena. No delta MS/LOC. No slurring of speech, unilateral weakness. No dysuria. Hospital Course: Patient had TTE performed on 7/24/91 which showed increase trans-mitral valve gradient. Given patient's history of MVR, TEE was recommended. On the night of 7/24/91, patient (who was known to sundown) tried to attack her roommate and fell on the floor hitting her head. No change in MS. In the setting of being on coumadin, patient had a stat CT of the head done which was negative for bleeds. On 7/25/91, patient went down for a TEE but desated to 80s and the procedure was aborted. Dr. Janssen then decided to perform cardiac cath on the patient Saturday 7/28/91. On 7/26/91, decision was made to put her on BIPAP for the night, and the patient did well and felt rested the next morning. On 7/27/91, because patient looked so well, she was not put on BIPAP but rather only on O2NC for the night. This AM, patient's respiratory status declined satting in the low 90s while on O2 6LNC. ABG was performed which showed 7.48/52/50. After speaking to cardiology, decision was made to transfer the patient to the CCU for intubation prior to cardiac cath this PM. A TEE will be performed while patient is intubated. Patient is currently NPO ready for cath this PM. PMH: 1.\tCOPD- PFTs on 7/17/91 w/ FEV1=32%, FEV1/FVC=63% 2.\tMVR- St. Jude's valve, 2085 3.\tHOCM- s/p myomectomy in 2085, TTE in 2090 showed EF=75%, no segmental wall motion abnormalities, dilated LA, mild AI; cardiac cath in 5/85 w/ only mild plaques 4.\tL CEA- 10/89 5.\tGI bleed- secondary to angiodysplasia, EGD/colonoscopy on 7/10/91 showed hiatal hernia and diverticulosis respectively 6.\tTrigeminal neuralgia Medications on Admission: Foradil inhaler Metoprolol 50mg bid Flovent 110mcg 4 puffs bid Nexium 40mg qd Verapamil SR 120mg qd MVI Atrovent inhaler Coumadin 2.5mg qd Serevent 1 puff bid Fosamax 70mg qweek Lescol 20mg qd Zyrtec 10mg qd prn Tums qd Albuterol inhaler Tegretol XR 400mg qd Iron 325mg qd Metamucil 1 tsp qd Enemas prn Baclofen 5mg tid Ultram 50mg prn Tylenol prn Neurontin 600mg tid Zithromax prior to dental work Medication on Transfer Heparin 1000U/hr IV in premixed continuous Laxis 20mg IV bid Prednisone 40mg po qam Carbamazepin 200mg po bid ASA 325mg po qd Atrovent 0.5mg neb qid Levofloxacin 500mg po qd Iron 325mg po qd Baclofen 5mg po tid Fosamax 70mg po qweek Lescol 20mg po qd Nexium 40mg po qd Verapamil sr 120mg po qd MVI 1tab po qd Serevent 1puff inh bid Lopressor 50mg po bid Flovent 110mcg inh bid Colace 100mg po tid Allergies: \t Amoxicillin, penicillin, Compazine, Bactrim, Sudafed, and Percodan. SH: \tLives alone, son lives in NM, come up to help out \tTob:\t\t50 pack year history, quit \tEtOH: \t\tDenies \tIllicits: \tDenies FH:\tNot elicited Exam: VS:\tT=97.8\t HR=83\tBP=106/58\tRR=24\t SaO2= 98% 6L General: in mild respiratory distress HEENT: NC/AT, PERRL, anicteric sclera. EOMI. OP w/ MMM, no erythema or injection. Skin:\tNo rashes Neck: Supple, full ROM, JVP flat (difficult to assess) Chest: Diffuse insp and exp wheeze CVS: \tTachy nl S1 S2, distorted by pulmonary sounds Abd: \t+BS, soft, ND/NT GU:\tDeferred Extr: 1+ pitting edema, warm extremities Neuro: Alert and oriented x 3 DATA: Chemistry:\t Plasma Sodium 140 135-145 mmol/L Plasma Potassium 3.4 3.4-4.8 mmol/L Plasma Chloride 95 L 100-108 mmol/L Plasma Carbon Dioxide 34.5 H 23.0-31.9 mmol/L Calcium 9.1 8.5-10.5 mg/dl Phosphorus 2.0 L 2.6-4.5 mg/dl Magnesium 1.4 1.4-2.0 meq/L Plasma Urea Nitrogen 29 H 8-25 mg/dl Plasma Creatinine 0.9 0.6-1.5 mg/dl ABG: ART FIO2 5LNC FIO2/L Result Text: .30 ART TEMP OUT 37.0 deg C Arterial pH 7.48 H 7.35-7.45 Arterial PCO2 52 H 35-42 mm/Hg Arterial PO2 50 L 80-100 mm/Hg Ionic Calcium 1.21 1.14-1.30 mmol/L Heme:\t WBC 9.9 4.5-11.0 th/cmm HCT 36.0 36.0-46.0 % HGB 12.1 12.0-16.0 gm/dl RBC 4.19 4.00-5.20 mil/cmm PLT 232 150-350 th/cumm MCV 86 80-100 fl MCH 29.0 26.0-34.0 pg/rbc MCHC 33.7 31.0-37.0 g/dl RDW 26.1 H 11.5-14.5 % PT 14.2 H 11.1-13.1 sec PT-INR 1.3 Result Text: PT-INR values are valid only for WARFARIN ANTI-COAG THERAPY. APTT 57.3 H 22.1-35.1 sec CT angiogram- Negative for PE on preliminary read CXR:\tSmall bilateral pleural effusions EKG:\tSinus tachycardia at 112bpm, LAE, 1mm ST depressions V4-V6 (similar when compared to previous EKG w/ tachycardia to 120) TTE: Compared to the report of 09/09/2090, the transmitral gradients have increased, the degree of MR and AI has increased, and the RV systolic pressure has increased significantly. LV size and function remain normal. If clinically indicated, a transesophageal echo is suggested to better evaluate the prosthetic valve. CTHead: \t1. NO ACUTE INTRACRANIAL HEMORRHAGE OR FRACTURE. \t2. NON-SPECIFIC PERIVENTRICULAR AND SUBCORTICAL WHITE MATTER \tHYPOATTENUATING AREAS ARE LIKELY DUE TO CHRONIC MICROANGIOPATHIC \tCHANGES. ___________________________________________ Impression: Ms. Mitchell is a 75 yo woman w/ multiple medical problems including HOCM, s/p MVR and myomectomy as well as COPD who presents with increasing SOB and DOE for the past four days. Of note, the patient notes increasing DOE for the past 4 months but with acute decompensation today. Her progressive DOE may be related to CHF, especially in the setting of ischemic changes seen on EKG however, her acute decompensation may be related to COPD exacerbation in the setting of a viral illness. Plan: SOB- Unclear etiology currently by may likely be multifactorial --Treat COPD w/ Atrovent nebs, prednisone, O2, Levaquin --Patient did not tolerate tachycardiac so Albuterol was DC'ed --Follow urine output secondary to IV Lasix --Daily weights, strict Is and Os --Keep O2 sats 88-90% EKG changes- Pt w/o known history of CAD however w/ and ST depressions on EKG that appear rate related --IV Lopressor to decrease rate --ASA --Cardiac monitor --Cardiac cath per Dr. Janssen with TEE MVR- On Coumadin --Follow INR Trigeminal neuralgia- Improving w/ pain control and Tegretol --Continue current management FEN- --Encourage Pos --Replete lytes as needed 6. Prophylaxis- Nexium, Coumadin _________________________________ Henry Norton, MD Pager #45074"
```

#### Output

```
"7/24/91 <DATE>",
"7/17/91 <DATE>",
"2085 <DATE>",
"5/85 <DATE>",
"6146161 <MEDICALRECORD>", "2091-07-28 <DATE>","7/25/91 <DATE>",
"2090 <DATE>",
"July 24, 2091 <DATE>",
"Janssen <DOCTOR>",
"Henry Norton <DOCTOR>",
"Geoffrey Lucas <DOCTOR>",
"Evelyn Conway <DOCTOR>",
"Saturday 7/28/91 <DATE>",
"NM <STATE>",
"7/27/91 <DATE>",
"7/10/91 <DATE>",
"7/26/91 <DATE>",
"Mitchell, Katie <PATIENT>",
"09/09/2090 <DATE>",
"Mitchell <PATIENT>",
"75 <AGE>",
"Raymond <CITY>",
"Jul 28, 2091 <DATE>",
"10/89 <DATE>",
"45074 <PHONE>"
```

### Explanation

As you can see, input and outputs are longer as shown in the example. For long input and output, there are some major problems with LLMs as listed below:

1. LLMs token size is limited, hence you cannot fit the whole input at once. So, large inputs are challenging.
2. LLMs miss some entity classes in their prediction although we specifically instruct them to predict those classes since there is large output label space.
3. LLMs start generating repetitive output since our output length is also very long.
4. Sometime LLMs get confused and swap the entity classes with each other.
5. On keeping other parameters constant, refreshing the page will give different output (wrong output) --> Not reliable.

Tackling these problems and building efficient biomedical system is an essential task.

## Literature to Read

1. [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
2. [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)
3. [Thinking about GPT-3 In-Context Learning for Biomedical IE? Think Again](https://arxiv.org/abs/2203.08410)
4. [KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction](https://arxiv.org/abs/2104.07650)

Note: If you do not have any basic idea about prompting, then read Sections 2,4,6 and 7 from [Pre-train, Prompt, and Predict](https://arxiv.org/abs/2107.13586)

## Steps to Perform Task

Use this to save your JSON file - `json.dumps(your_json_string, indent=4, ensure_ascii=False)`

#### Task 1 (Tentative Deadline - Oct. 21, 2022)

1. Select datasets and assign your name to them.
2. Get GPT-3 free version.
3. Use 5 samples evaluation/test set from your datasets, and use GPT-3 to solve those samples.
4. Based on GPT-3 solution/output, analyze/identify the above-listed issues that your dataset is facing.
5. If you encounter a new issue (not one of the given four issues), then add that issue to the list.
6. You need to make folder with your name, sub-folder with dataset name and commit jsonl file with five instances in below format -

```
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "new_issues": lst}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "new_issues": lst}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "new_issues": lst}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "new_issues": lst}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "new_issues": lst}
```

#### Task 2 (Tentative Deadline - Oct. 31, 2022)

1. Now use the examples that you selected in Task 1 and see the issues that you have got.
2. Choose any 2 prompting techniques according to the issue that you want to solve. For example, "Chain of Thought" or "Least-to-Most". You can go and search for literature and choose techniques from them.
3. Commit JSON file in following format for those two prompting techniques that you chose:

```json
{
"technique_1": {
  "explaination": "[2-3] line description"
  "referece_papers": "[List of links]"
}
"technique_2": {
  "explaination": "[2-3] line description"
  "referece_papers": "[List of links]"
}
}
```

4. Now apply those techniques to solve issue in those 5 examples and see improvement.
5. Commit jsonl file with five instances in below format and describe what kind of improvement you got -

```
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "improvement": str}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "improvement": str}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "improvement": str}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "improvement": str}
{"dataset_name": str, "input": str or lst, "output": str or lst, "known_issues": lst, "improvement": str}
```

#### Task 3 (Tentative Deadline - Nov. 13, 2022)

1. We want you to explore prompting techniques that you developed in Task 2 with different models. We want to try below models:

- GPT-2: https://huggingface.co/gpt2?text=My+name+is+Mariama%2C+my+favorite
- MetaICL: https://github.com/facebookresearch/MetaICL
- BioGPT: https://github.com/microsoft/BioGPT
- T0: https://huggingface.co/bigscience/T0pp
- PaLM: https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html
- In-BoXBART: https://huggingface.co/cogint/in-boxbart

2. You need to select any two models from the above model list and implement them for your dataset.
3. You need to perform 2 experiments with those models:

- Evaluate the model directly on test data without any prompting
- Use the best prompting technique from Task 2 and evaluate the model with that prompting technique

4. If it's a classification task, then report your accuracy on the above two experiments. But if it's a generation task, then report the Rouge-L metric on the above experiments.
5. Commit your evaluation script with two models (.py or .ipynb) and a text file with performance metrics.

(You can use Agave or Google colab to do experiments).

#### Task 4 (Tentative Deadline - Nov. 27, 2022)

[TBD] - task 4 will be some analysis.

#### Broader level idea of project work

1. We are going to use GPT-3 (free version) for some analysis. You are going to try out 15-20 examples from datasets assigned to you and observe the above problems or any other specific problem you encounter. If you encounter a new problem, then you need to mention that.
2. You are going to try 2 prompting techniques to overcome all or some issues that we found in step (1).
3. Now, you are going to extend step (2) prompting methods to different models such as BART, and T5.
4. Figure out if we need any fine-tuning with your new prompting technique.
5. Evaluate your technique on the evaluation set from respective datasets.

## Datasets

Please pick the dataset you like from the below list and add your name corresponding to that dataset.

- See dataset details here - [n2c2 datasets](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
- BioASQ 8 - [see here](http://www.bioasq.org/participate/challenges_year_8)
- POS Tagging - [see here](http://www.nactem.ac.uk/GENIA/tagger/)
- CRAFT NER - [see here](https://bionlp-corpora.sourceforge.net/CRAFT/)

| Dataset Name             | Assignee | Issues |
| ------------------------ | -------- | ------ |
| Adverse Drug Events 2018 | Varun    |        |
| CRAFT NER                | Varun    |        |
