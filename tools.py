# 字符串数字序列转数字list并充填到max_length
def strnum2list(text,max_length):
    text = text.replace("\n", "")
    text_array = []
    if len(text)<max_length:
        for c in text:
            text_array.append(int(c))
        text_array = text_array + [0]*(max_length-len(text))
    else:
        for i in range(max_length):
            text_array.append(int(text[i]))
    return text_array