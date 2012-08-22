/**
 * This program is used by <i>Hierarchical Sampling for Multi-Instance Ensemble Learning</i>.
 * 
 */
package moe.mimethods;

import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Debug.Random;

/**
 * Class for TMIE.
 * 
 * @author Meng Fang (Meng.Fang@student.uts.edu.au)
 * @version $Revision: 2.0 $
 */
public class TEnsembleMI {
	public Vote tEnsemble;
	
	int sizeOfEnsemble = 1;
	
	public TEnsembleMI(String nameOfBaseClassifier, int tInterSampling){
		sizeOfEnsemble = tInterSampling;
		tEnsemble = new Vote();
		Classifier[] clset = new Classifier[sizeOfEnsemble];
		for(int i=0; i < sizeOfEnsemble; i++){
			try {
				clset[i] = (Classifier)Class.forName(nameOfBaseClassifier).newInstance();
			} catch (InstantiationException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		// average 
		SelectedTag st = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
		
		tEnsemble.setCombinationRule(st);
		tEnsemble.setClassifiers(clset);
		
		tEnsemble.setSeed(2);
	}
	
	public void trainSingleCL(int indexOfcl, Instances insts) throws Exception{
		this.tEnsemble.getClassifier(indexOfcl).buildClassifier(insts);
	}
	
	public void trainAll(Instances insts) throws Exception{
		Instances original = insts;

		Random rd = new Random();
		
		for(int i=0; i < this.sizeOfEnsemble; i++){
			rd.setSeed(System.currentTimeMillis());
			Instances newdata = original.resample(rd);
			this.tEnsemble.getClassifier(i).buildClassifier(newdata);
		}
	}
	
	public double predictInst(Instance inst) throws Exception{
		double pred = -1;
		
		pred = this.tEnsemble.classifyInstance(inst);
		return pred;
	}
	
	public double predictAll(Instances insts) throws Exception {
		int sum = insts.numInstances();
		double pred = 0.0;
		int correct = 0;
		
		for(int i=0; i < sum; i++) {
			double temp = predictInst(insts.instance(i));
			if(temp == insts.instance(i).classValue())
				correct++;
		}
		
		pred = 1.0*correct/sum;
		
		return pred;
	}
}
