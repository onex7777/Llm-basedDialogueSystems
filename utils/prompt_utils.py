# encoding=utf-8
import json

from scene_config import scene_prompts
from utils.date_utils import get_current_date
from utils.helpers import get_slot_query_user_json, get_slot_update_json


def get_slot_update_message(scene_name, dynamic_example, slot_template, user_input):
    message = scene_prompts.slot_update.format(scene_name, get_current_date(), dynamic_example, json.dumps(get_slot_update_json(slot_template), ensure_ascii=False), user_input)
    return message


def get_slot_query_user_message(scene_name, slot, user_input):
    """
    数用于生成询问用户特定槽位信息的消息。函数参数包括场景名称scene_name、槽位名称slot和用户输入user_input。函数内部通过调用get_slot_query_user_json函数获取槽位的查询信息，并将其与场景名称和用户输入一起格式化为一条消息。最后返回该消息。
    """
    message = scene_prompts.slot_query_user.format(scene_name, json.dumps(get_slot_query_user_json(slot), ensure_ascii=False), user_input)
    return message
