import argparse,json
from .api import MikteengAI

def main():
    parser=argparse.ArgumentParser(prog='mikteeng-ai');parser.add_argument('model');parser.add_argument('text');parser.add_argument('--task',choices=['generate','roles','sentence','passage'],default='generate');parser.add_argument('--action');args=parser.parse_args();model=MikteengAI.load(args.model)
    if args.task=='generate':result=model.generate(args.text)
    elif args.task=='roles':result=model.predict_roles(args.text)
    elif args.task=='sentence':result=model.predict(args.text)
    else:
        if not args.action:parser.error('--action is required for passage predictions')
        result=model.answer_passage(args.text,args.action)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
