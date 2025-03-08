from autogen import UserProxyAgent, config_list_from_json 
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent
from dotenv import load_dotenv

load_dotenv()

filter_dict = {
    "model": ["gpt-4"]
}
config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.txt", filter_dict=filter_dict
)
llm_config = {"config_list": config_list, "timeout": 120}

teachable_agent = ConversableAgent(
    name="teachable_agent", llm_config=llm_config
)

teachability = Teachability(
    reset_db=False,
    path_to_db_dir="./tmp/user",
)

teachability.add_to_agent(teachable_agent)

user = UserProxyAgent("user", human_input_mode="ALWAYS")
teachable_agent.get_human_input(
    prompt="""
You are assistant that create a Course Learning Outcome (CLO) from the input.
Input include:
- Program Learning Outcome (PLO)
- Book Title
- Book Author
- Table of Content
Requirement:
- Write in Vietnamese
- Output include only the Course Learning Outcome (CLO)
"""
)
teachable_agent.initiate_chat(
    user, message="""
PLO:
1. Có khả năng áp dụng kiến thức về Toán và khoa học cơ bản để giải quyết và nghiên cứu các vấn đề trong lĩnh vực Công nghệ thông tin.
2. Có khả năng áp dụng các nguyên lý trong tính toán và lập trình cũng như trong các ngành liên quan để phân tích vấn đề, thiết kế, thực hiện và đánh giá các giải pháp Công nghệ thông tin.
3. Có khả năng tư duy phản biện, tư duy sáng tạo, tư duy khởi nghiệp, ứng xử chuyên nghiệp; có đạo đức, trách nhiệm nghề nghiệp.
4. Có khả năng làm việc nhóm; có khả năng giao tiếp hiệu quả trong môi trường quốc tế và đa văn hóa; có trình độ ngoại ngữ TOEIC 450 trở lên hoặc tương đương.
5. Có khả năng hình thành ý tưởng, lựa chọn, thiết kế, tích hợp, đánh giá và quản trị hệ thống công nghệ thông tin trong bối cảnh doanh nghiệp, xã hội và môi trường.
6. Có khả năng lập kế hoạch và quản lý dự án công nghệ thông tin.
Title: Financial Machine Learning 
Author: Bryan Kelly, Dacheng Xiu
Table of Content:
Introduction: The Case for Financial Machine Learning 2
1.1 Prices are Predictions 2
1.2 Information Sets are Large 3
1.3 Functional Forms are Ambiguous 5
1.4 Machine Learning versus Econometrics 6
1.5 Challenges of Applying Machine Learning in Finance (and
the Benefits of Economic Structure) 9
1.6 Economic Content (Two Cultures of Financial Economics) 10
2The Virtues of Complex Models 15
2.1Tools For Analyzing Machine Learning Models 16
2.2Bigger Is Often Better . 20
2.3The Complexity Wedge 26
3 Return Prediction 28
3.1 Data . 31
3.2 Experimental Design 32
3.3 A Benchmark: Simple Linear Models
3.4 Penalized Linear Models 40
3.5 Dimension Reduction. 44
3.6 Decision Trees 53
3.7 Vanilla Neural Networks. 59
Electronic copy available at: https://ssrn.com/abstract=4501707
3.8 Comparative Analyses 64
3.9 More Sophisticated Neural Networks 68
3.10 Return Prediction Models For "Alternative" Data 70
4Risk-Return Tradeoffs 79
4.1 APT Foundations 79
4.2 Unconditional Factor Models 80
4.3 Conditional Factor Models 88
4.4Complex Factor Models 94
4.5High-frequency Models 96
4.6Alphas 98
5 Optimal Portfolios 105
5.1"Plug-in" Portfolios 107
5.2 Integrated Estimation and Optimization 111
5.3Maximum Sharpe Ratio Regression 112
5.4 High Complexity MSRR 116
5.5 SDF Estimation and Portfolio Choice 118
5.6Trading Costs and Reinforcement Learning 126
6Conclusions 132
References 134
Electronic copy available at: https://ssrn.com/abstract=4501707
"""
)