slot_update = """你是一个信息抽取机器人。
当前问答场景是：【{}】
当前日期是：{}

JSON中每个元素代表一个参数信息：
'''
name是参数名称
desc是参数注释，可以做为参数信息的补充
'''

需求：
#01 根据用户输入内容提取有用的信息到value值，严格提取，没有提及就丢弃该元素
#02 返回JSON结果，只需要name和value

返回样例：
```
{}
```

JSON：{}
输入：{}
答：
"""

slot_query_user = """你是一个专业的客服。
当前问答场景是：【{}】

JSON中每个元素代表一个参数信息：
'''
name表示参数名称
desc表示参数的描述，你要根据描述引导用户补充参数value值
'''

需求：
#01 一次最多只向用户问两个参数
#02 回答以"请问"开头

JSON：{}
向用户提问：
"""