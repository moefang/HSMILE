/**
 * This program is used by <i>Hierarchical Sampling for Multi-Instance Ensemble Learning</i>.
 * 
 */
package moe.mimethods;

import moe.sampling.MISamplingStrategy;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Class for HSMILE.
 * 
 * @author Meng Fang (Meng.Fang@student.uts.edu.au)
 * @version $Revision: 2.0 $
 */
public class HEnsembleMI {
	
	public Vote dEnsembleMI;
	
	int tInter =1;
	int tIntra =1;
	
	public HEnsembleMI(String nameOfBaseClassifier, int initialtInter, 
			int initialtIntra){
		this.tInter = initialtInter;
		this.tIntra = initialtIntra;
		int sizeOfEnsemble = tInter*tIntra;
		
		dEnsembleMI = new Vote();
				
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
		
		dEnsembleMI.setCombinationRule(st);
		dEnsembleMI.setClassifiers(clset);
		
		dEnsembleMI.setSeed(2);
	}
	
	public void train(int indexOfcl, Instances insts) throws Exception{
		this.dEnsembleMI.getClassifier(indexOfcl).buildClassifier(insts);
	}
	
	public void trainAll(Instances insts, MISamplingStrategy mis) throws Exception{
		
		int clIndex = 0;
		
//		if(tInter==1 && tIntra ==1) {
//		}
		
		for(int i=0; i < tInter; i++){
			for(int j=0; j < tIntra; j++){
				Instances samplingData = mis.DualSampling(insts);
				this.dEnsembleMI.getClassifier(clIndex).buildClassifier(samplingData);
				clIndex++;
			}
		}
	}
	
	public void trainAllWithNegativeOnlySampling(Instances insts, MISamplingStrategy mis) throws Exception{
		
		int clIndex = 0;
		
		for(int i=0; i < tInter; i++){
			for(int j=0; j < tIntra; j++){
				Instances samplingData = mis.DualSamplingWithNegativeOnly(insts);
				this.dEnsembleMI.getClassifier(clIndex).buildClassifier(samplingData);
				clIndex++;
			}
		}
	}
	
	public void trainAllWithNaiveTotalSampling(Instances insts, MISamplingStrategy mis) throws Exception{
		int clIndex =0;
		for(int i=0; i < tInter; i++){
			for(int j=0; j < tIntra; j++) {
				Instances samplingData = mis.DualSamplingWithNaiveTotalSampling(insts);
				this.dEnsembleMI.getClassifier(clIndex).buildClassifier(samplingData);
				clIndex++;
			}
		}
	}
	
	public double predictInst(Instance inst) throws Exception{
		double pred = -1;
		
		pred = this.dEnsembleMI.classifyInstance(inst);
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
