# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys




from paddlenlp import SimpleServer, Taskflow

# The schema changed to your defined schema
# schema = ["开票日期", "名称", "纳税人识别号", "开户行及账号", "金额", "价税合计", "No", "税率", "地址、电话", "税额"]
schema = [
    '姓名', '联系人姓名', '进修医师', '质控护士',
    '住院次数', '性别', '出生日期', '年龄', '国籍', '民族', '出生地',
    '籍贯', '职业', '婚姻', '现住址', '电话', '户口地址', '工作单位及地址', '单位电话',
    '地址', '入院时间', '入院科别', '出院时间', '出院科别', '实际住院时间', '门（急）诊诊断', '疾病编码',
    '损伤、中毒的外部原因', '病理诊断', '质控日期', '病案质量',
    '就诊号', '健康卡号', '登记号', '病案号', '身份证号', '病理号', '科主任', '主任（副主任）医师','主治医师', '住院医师', '责任护士', '实习医师', '编码员', '质控医师',  "出院诊断-疾病名称","出院诊断-疾病编码"
]
# The task path changed to your best model path
model_pth = ".\\model_best\\model_best_with_tb\\"

uie = Taskflow(
    "information_extraction",
    model="global-pointer", 
    task_path=model_pth,
)
# If you want to define the finetuned uie service
app = SimpleServer()
app.register_taskflow("taskflow/uie", uie)
