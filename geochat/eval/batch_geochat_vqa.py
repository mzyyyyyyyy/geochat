import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    
    questions=[]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

        
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    
    for i in tqdm(range(0,len(questions),args.batch_size)):
        input_batch=[]
        input_image_batch=[]
        count=i
        image_folder=[] 
        batch_end = min(i + args.batch_size, len(questions))

        # 1，读取一个 batch 内的所有 prompt 文本，并 token 化，存储到 input_batch；然后读取 batch 内对应的图像，存储到 image_folder。
        for j in range(i,batch_end):
            image_file=questions[j]['image'] # 这里的循环结构是沿着 jsonl 文件找图像，桥梁是图像文件名。
            qs=questions[j]['text']
            
            # 给图像加上标志'<image>'，用来帮助模型区分图像和文本。
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs 

            # 在构建多轮对话模型时（如聊天机器人或对话生成模型），这段代码用于构造输入提示，使模型能够根据上下文生成合理的回复。
            conv = conv_templates[args.conv_mode].copy() # 选择对话模板
            conv.append_message(conv.roles[0], qs) # 构建对话内容：conv.roles[0] 表示第一个角色（通常是“用户”或“人类”）。qs 是要追加的消息内容（例如，用户的输入问题）。
            conv.append_message(conv.roles[1], None) # conv.roles[1] 表示第二个角色（通常是“模型”或“GPT”）。None 作为消息内容追加到对话中，这意味着还没有实际的消息文本（模型的回复尚未生成）。
            prompt = conv.get_prompt() # 生成对话提示，prompt 是一个字符串 

            # 将 prompt(文本) tokenizer，并将 batch 内的所有 tokenized prompt 组织到一起。
            # 具体是怎么做的呢？？？
            # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) # shape = (1, 62)
            input_batch.append(input_ids)

            # 读取 prompt 对应的图像
            image = Image.open(os.path.join(args.image_folder, image_file)) 
            image_folder.append(image)

            # 这段代码的核心目的是设置文本生成的停止条件，使模型在生成到特定分隔符时停止，从而保证生成的输出符合预期的格式或内容长度要求。暂不care。
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids) 

        # 2，这段代码的目的是对1个 batch 的 prompt 进行 padding，以使所有输入张量的长度相同，然后将它们 concatenate 成一个大的张量。
        max_length = max(tensor.size(1) for tensor in input_batch) # batch 中最长的 prompt 的长度.
        # final_input_list = [torch.cat((torch.zeros((1,max_length - tensor.size(1)), dtype=tensor.dtype,device=tensor.get_device()), tensor),dim=1) for tensor in input_batch]
        final_input_list = [torch.cat((torch.zeros((1, max_length - tensor.size(1)), dtype=tensor.dtype, device=tensor.device if tensor.is_cuda else 'cpu'), tensor), dim=1) for tensor in input_batch]
        final_input_tensors=torch.cat(final_input_list,dim=0) # 应该是把 batch 内的 prompt 合起来了，然后一起转移到 GPU 上了。shape = (1, 62)
        
        # 3，图像预处理：该函数被设计用于深度学习中的图像预处理，特别适合在输入图像到模型前进行必要的调整和转换，确保输入数据符合模型的预期格式和要求。这里并没有进行图像编码。
        image_tensor_batch = image_processor.preprocess(image_folder,crop_size ={'height': 504, 'width': 504},size = {'shortest_edge': 504}, return_tensors='pt', padding=True)['pixel_values']
        # shape = (1, 3, 504, 504)

        # 4，推理：将2，3的结果输入给 model，输出结果。
        with torch.inference_mode():
            # output_ids = model.generate( final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=256,length_penalty=2.0, use_cache=True)
            model = model.to(device='cpu', dtype=torch.float32)
            output_ids = model.generate( final_input_tensors, images=image_tensor_batch, do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=256,length_penalty=2.0, use_cache=True) # 前面的 prompt, 编码器处理过的 image 都输入到这里的 model 里去了，最终，shape = (1, 64)
            # 前面的 prompt, 编码器处理过的 image 都输入到这里的 model 里去了，最终，shape = (1, 64) # num_beams = 1 表示在解码过程中使用贪心搜索算法：贪心搜索是一种简单的解码策略。在每个时间步，模型根据当前的概率分布，选择概率最高的下一个标记。

        # 它用于确保输出序列的前几位（与输入长度相同）和输入序列是否一致。如果它们不同，可能表明生成的输出有问题，或者模型没有正确复现输入。
        # 具体为什么，暂略。
        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        # 5，这行代码将模型生成的 token（不包括与输入相同的部分）解码为文本，同时忽略掉不必要的特殊标记，从而得到干净的输出。
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        
        # 6，输出结果
        for k in range(0,len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            ans_id = shortuuid.uuid()
            
            ans_file.write(json.dumps({
                                    "question_id": questions[count]["question_id"],
                                    "image_id": questions[count]["image"],
                                    "answer": output,
                                    }) + "\n")
            count=count+1
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
