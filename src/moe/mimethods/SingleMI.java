/**
 * This program is used by <i>Hierarchical Sampling for Multi-Instance Ensemble Learning</i>.
 * 
 */
package moe.mimethods;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class for SMI.
 * 
 * @author Meng Fang (Meng.Fang@student.uts.edu.au)
 * @version $Revision: 2.0 $
 */
public class SingleMI {
	
	public Classifier cl;
	
	public SingleMI(String nameOfBaseClassifier){
		
		try {
			this.cl = (Classifier)Class.forName(nameOfBaseClassifier).newInstance();
			
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
	
	public void train(Instances insts) throws Exception{
		
		this.cl.buildClassifier(insts);
	}
	
	public double predictInst(Instance inst) throws Exception{
		double pred = -1;
		
		pred = this.cl.classifyInstance(inst);
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
