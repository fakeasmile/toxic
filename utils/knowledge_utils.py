import json
from pathlib import Path


class HomophoneRestorer:
    def __init__(self, homo_dict_path=None):
        self.homo_map = {}
        if homo_dict_path and Path(homo_dict_path).exists():
            with open(homo_dict_path, "r", encoding="utf-8") as f:
                self.homo_map = json.load(f)

    def restore(self, text):
        if not self.homo_map:
            return text
        restored = text
        for variant, original in self.homo_map.items():
            restored = restored.replace(variant, original)
        return restored


class CodedTermMatcher:
    def __init__(self, coded_terms_path=None, max_terms=200):
        self.term_list = []
        self.term_to_id = {}
        self.max_terms = max_terms
        if coded_terms_path and Path(coded_terms_path).exists():
            with open(coded_terms_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.term_list = data[:max_terms]
            elif isinstance(data, dict):
                self.term_list = list(data.keys())[:max_terms]
            for i, term in enumerate(self.term_list):
                self.term_to_id[term] = i + 1

    def match(self, text):
        matched_ids = []
        for term, tid in self.term_to_id.items():
            if term in text:
                matched_ids.append(tid)
        return matched_ids

    @property
    def num_terms(self):
        return min(len(self.term_list) + 1, self.max_terms + 1)


DEFAULT_HOMOPHONE_MAP = {
    "伞兵": "傻逼", "纱壁": "傻逼", "莎比": "傻逼", "sb": "傻逼", "SB": "傻逼",
    "nc": "脑残", "NC": "脑残",
    "fw": "废物", "FW": "废物",
    "qs": "去死", "QS": "去死",
    "xfh": "小粉红",
    "nmsl": "你妈死了", "NMSL": "你妈死了",
    "坦克": "胖女人", "小仙女": "田园女权",
    "普信男": "普通且自信的男性",
    "润": "移民逃跑", "卷": "过度竞争",
    "乐色": "垃圾", "辣鸡": "垃圾",
    "蛙蛙": "台湾人蔑称", "绿蛙": "台独分子蔑称",
    "阿三": "印度人蔑称", "黑鬼": "黑人蔑称",
    "白左": "西方左派蔑称", "公知": "公共知识分子贬称",
    "五毛": "网络评论员贬称", "美分": "亲美人士贬称",
    "粉红": "民族主义者贬称", "皇汉": "大汉族主义者",
    "绿茶": "虚伪女性贬称", "海王": "花心男性贬称",
    "舔狗": "卑微追求者贬称", "直男": "缺乏情商男性贬称",
    "娘炮": "女性化男性贬称", "凤凰男": "农村出身男性贬称",
    "扶弟魔": "过度帮扶弟弟的女性贬称", "田园女权": "伪女权贬称",
}

DEFAULT_CODED_TERMS = [
    "傻逼", "脑残", "废物", "去死", "垃圾",
    "伞兵", "纱壁", "莎比", "乐色", "辣鸡",
    "坦克", "小仙女", "普信男", "绿茶", "舔狗",
    "娘炮", "凤凰男", "田园女权", "扶弟魔",
    "黑鬼", "阿三", "白左", "公知", "五毛", "美分",
    "粉红", "皇汉", "蛙蛙", "绿蛙",
    "nmsl", "sb", "nc", "fw", "qs", "xfh",
    "婊子", "贱人", "骚货", "荡妇", "破鞋",
    "狗日的", "畜生", "禽兽", "杂种", "王八蛋",
    "滚蛋", "闭嘴", "去你的", "少废话",
    "种族歧视", "性别歧视", "地域歧视",
    "黑猪", "白猪", "黄猴",
    "支那", "蝗虫", "蛆虫",
    "呵呵", "就这", "不会吧",
    "带节奏", "引战", "钓鱼", "阴阳怪气",
    "扣帽子", "站队", "捧杀",
    "非人化", "物化", "标签化",
]


def build_default_homophone_map(output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_HOMOPHONE_MAP, f, ensure_ascii=False, indent=2)
    print(f"默认谐音映射已保存到: {output_path}")


def build_default_coded_terms(output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CODED_TERMS, f, ensure_ascii=False, indent=2)
    print(f"默认编码术语词表已保存到: {output_path}")


def get_platform_id(platform_str):
    if platform_str == "zhihu":
        return 0
    elif platform_str == "tieba":
        return 1
    else:
        return 2


def get_topic_id(topic_str):
    topic_map = {"race": 0, "gender": 1, "region": 2, "lgbt": 3, "none": 4}
    return topic_map.get(topic_str, 4)
