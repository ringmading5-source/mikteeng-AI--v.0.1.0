"""One-time converter for trusted experimental checkpoints; not a serving API."""
import importlib,pickle,sys
from pathlib import Path
from mikteeng_ai import MikteengAI
MODULES={'generator','sentence_generator','subject_generator','revisable_roles','roles','predictor','optimized','features','learner','system','generative_system'}
class LegacyLoader(pickle.Unpickler):
    def find_class(self,module,name):
        if module in MODULES:return getattr(importlib.import_module('mikteeng_ai._models.'+module),name)
        return super().find_class(module,name)
def migrate(source,destination):
    with open(source,'rb') as stream:legacy=LegacyLoader(stream).load()
    model=MikteengAI();model.generator=legacy.generator;model.roles=legacy.generator.roles;model.passage_model=legacy.previous.passage;model.sentence_model=legacy.previous.previous;model.save(destination)
    return model
if __name__=='__main__':migrate(sys.argv[1],sys.argv[2])
