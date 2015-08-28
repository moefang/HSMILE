/**
 * This program is used by <i>Hierarchical Sampling for Multi-Instance Ensemble Learning</i>.
 * 
 */
package moe.test;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

import moe.mimethods.HEnsembleMI;
import moe.mimethods.SingleMI;
import moe.mimethods.TEnsembleMI;
import moe.sampling.MISamplingStrategy;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PropositionalToMultiInstance;

/**
 * Class for entrance of this program.
 * 
 * @author Meng Fang (Meng.Fang@student.uts.edu.au)
 * @version $Revision: 2.0 $
 */
public class MainRun {
	
	public void runData(String inputFile, String resultFile,String nameOfClassifier, int tInter, int tIntra, double miu){
		FileReader fr;
		try {
			fr = new FileReader(inputFile);
			Instances propositionalData = null; 
			propositionalData = new Instances(fr);
			fr.close();
			
			propositionalData.setClassIndex(propositionalData.numAttributes()-1);
			
			PropositionalToMultiInstance trans=new PropositionalToMultiInstance();
			trans.setInputFormat(propositionalData);
			Instances miData= Filter.useFilter(propositionalData, trans);
			
			CrossValidation cv = new CrossValidation(miData, 10);
			
			PrintWriter pw;
			pw = new PrintWriter(resultFile);
				
			for(int i=0; i< 10; i++){
				
				System.out.println("cv-"+ (i+1));
				System.out.println("  train ...");
				Instances train = cv.getTrainSubset(i);
				Instances test = cv.getTestSubset(i);
				
				HEnsembleMI dsmile = new HEnsembleMI(nameOfClassifier,tInter,tIntra);
				
				MISamplingStrategy mis = new MISamplingStrategy(miData,miu);
				
				dsmile.trainAll(train, mis);
				
				System.out.println("  test ...");
				double result = dsmile.predictAll(test);
				
				System.out.println(result);
				pw.println(i+","+result);
				
				System.out.println();
			}
			pw.flush();
			pw.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		
		// an example for running Musk1 data set.
		MainRun mr = new MainRun();
		String inputFile = "musk1.arff";
		String resultFile ="musk1mioptball.csv";
		String nameOfClassifier = "weka.classifiers.mi.MIOptimalBall";
		int tInter = 2;
		int tIntra = 3;
		double miu = 4.5;
		
		mr.runData(inputFile, resultFile, nameOfClassifier, tInter, tIntra, miu);
		
		System.out.println("****** Good Luck :)******");
	}
}
