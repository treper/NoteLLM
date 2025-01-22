import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf8",line_buffering=True)
import csv
import tqdm
import argparse
from statistics import mean
import sys
import jieba
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction



global sm_func
sm_func = {}
sm_func["None"] = None
sm_func["sm1"] = SmoothingFunction().method1
sm_func["sm2"] = SmoothingFunction().method2
sm_func["sm3"] = SmoothingFunction().method3
sm_func["sm4"] = SmoothingFunction().method4
sm_func["sm5"] = SmoothingFunction().method5
sm_func["sm6"] = SmoothingFunction().method6
sm_func["sm7"] = SmoothingFunction().method7

def read_answer(answer_path):
    predicts = []
    answers = []
    with open(answer_path,'r') as fp:
        reader = csv.reader( (line.replace('\0','') for line in fp) )
        for ind, data in enumerate(tqdm.tqdm(reader)):
            predict = data[0]
            answer = data[1]
            predicts.append(predict)
            answers.append(answer)
    return (predicts, answers)

class BLEU(object):
    

    def compute_bleu(self, refs, systems, sm):
        bleu_1 = []
        bleu_2 = []
        bleu_3 = []
        bleu_4 = []
        bleu_all = []
        rouge1 = []
        rouge2 = []
        rougel = []
        cnt = 0
        rouger = Rouge()
        for i in tqdm.tqdm(range(len(systems))):
            if refs[i].strip()=='':
                continue
            if systems[i].strip()=='':
                rouge1.append(0)
                rouge2.append(0)
                rougel.append(0)
                bleu_1.append(0)
                bleu_2.append(0)
                bleu_3.append(0)
                bleu_4.append(0)
                bleu_all.append(0)
            else:
                try:
                    scores = rouger.get_scores(systems[i], refs[i])
                    rouge1.append(scores[0]['rouge-1']['f'])
                    rouge2.append(scores[0]['rouge-2']['f'])
                    rougel.append(scores[0]['rouge-l']['f'])
                except:
                    rouge1.append(0)
                    rouge2.append(0)
                    rougel.append(0)
                refs[i] = refs[i].split()
                systems[i] = systems[i].split()
                B1 = sentence_bleu(refs[i], systems[i], weights = (1, 0, 0, 0), smoothing_function=sm)
                bleu_1.append(float(B1))
                B2 = sentence_bleu(refs[i], systems[i], weights = (0, 1, 0, 0), smoothing_function=sm)
                bleu_2.append(float(B2))
                B3 = sentence_bleu(refs[i], systems[i], weights = (0, 0, 1, 0), smoothing_function=sm)
                bleu_3.append(float(B3))
                B4 = sentence_bleu(refs[i], systems[i], weights = (0, 0, 0, 1), smoothing_function=sm)
                bleu_4.append(float(B4))    
                BA = sentence_bleu(refs[i], systems[i], smoothing_function=sm)
                bleu_all.append(float(BA))   
                cnt+=1

        return mean(bleu_1), mean(bleu_2), mean(bleu_3), mean(bleu_4), mean(bleu_all), mean(rouge1), mean(rouge2), mean(rougel)

    def print_score(self, ref_corpus, gen_corpus, sm):
        
        b1= corpus_bleu(ref_corpus, gen_corpus, weights = (1, 0, 0, 0), smoothing_function=sm)
        b2= corpus_bleu(ref_corpus, gen_corpus, weights = (0, 1, 0, 0), smoothing_function=sm)
        b3 = corpus_bleu(ref_corpus, gen_corpus, weights = (0, 0, 1, 0), smoothing_function=sm)
        b4 = corpus_bleu(ref_corpus, gen_corpus, weights = (0, 0, 0, 1), smoothing_function=sm)
        ba = corpus_bleu(ref_corpus, gen_corpus, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function=sm)
        print("------------------------------------------")
        print(" BLEU-ALL: %02.3f, BLEU-1: %02.3f, BLEU-2: %02.3f, BLEU-3: %02.3f, BLEU-4: %02.3f" \
            %(ba * 100, b1 * 100, b2 * 100, b3 * 100, b4 * 100))


def compute_answer(answer,predict,save_path,type):
    all_nid = len(predict)
    print(f'answer size: {len(answer)}')
    print(f'predict size: {len(predict)}')
    bleu = BLEU()
    gen = []
    ref = []
    for i in range(len(answer)):
        gen.append(predict[i])
        ref.append(answer[i])
    sm = "sm1"
    gen = [' '.join(jieba.cut(item)) for item in gen]
    ref = [' '.join(jieba.cut(item)) for item in ref]
    b1, b2, b3, b4, ba, r1, r2, rl = bleu.compute_bleu(ref, gen , sm_func[sm])
    
    with open(os.path.join(save_path,f'{type}_result.txt'), 'w', newline='') as f:
        f.write("datalen: %d, ROUGE-1: %02.5f, ROUGE-2: %02.5f, ROUGE-L: %02.5f, S_FUNC: %s, BLEU-ALL: %02.5f, BLEU-1: %02.5f, BLEU-2: %02.5f, BLEU-3: %02.5f, BLEU-4: %02.5f" \
        %(len(gen),r1*100,r2*100,rl*100,sm, ba * 100, b1 * 100, b2 * 100, b3 * 100, b4 * 100))
        print("datalen: %d, ROUGE-1: %02.5f, ROUGE-2: %02.5f, ROUGE-L: %02.5f, S_FUNC: %s, BLEU-ALL: %02.5f, BLEU-1: %02.5f, BLEU-2: %02.5f, BLEU-3: %02.5f, BLEU-4: %02.5f" \
        %(len(gen),r1*100,r2*100,rl*100,sm, ba * 100, b1 * 100, b2 * 100, b3 * 100, b4 * 100))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--save_path", type=str)
    arg_parser.add_argument("--answer_path", type=str)
    arg_parser.add_argument("--type", type=str, default="category")

    args = arg_parser.parse_args()

    predicts, answers = read_answer(args.answer_path)
    compute_answer(answers,predicts,args.save_path,args.type)
    
