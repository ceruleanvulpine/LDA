import org.ejml.data.DenseMatrix64F;


public class mainmethod {

	public static void main(String[] args){
		
		DenseMatrix64F cov = new DenseMatrix64F(3,3);
		cov.set(0,0,30);
		cov.set(1,1,30);
		cov.set(2,2,30);

		DenseMatrix64F means = new DenseMatrix64F(3,3);
		means.set(0,0,10);
		means.set(1,1,10);
		means.set(2,2,10);
		
		double[] probs = new double[3];
		probs[0] = .3;
		probs[1] = .3; 
		probs[2] = .4;
		
		LdaData l = new LdaData(cov, means, probs);
		DenseMatrix64F[] data = l.genData(1000);
		System.out.println(data[0]);
		System.out.println(data[1]);
		System.out.println(data[2]);
		
		LDA l1 = new LDA(data);
		
		double[] coeff = l1.hypcoeff(0, 1);
		System.out.println(coeff[0]);
		System.out.println(coeff[1]);
		System.out.println(coeff[2]);
		System.out.println(coeff[3]);
		
		System.out.println();
		
		coeff = l1.hypcoeff(0, 2);
		System.out.println(coeff[0]);
		System.out.println(coeff[1]);
		System.out.println(coeff[2]);
		System.out.println(coeff[3]);
		
		System.out.println();
		
		coeff = l1.hypcoeff(1, 2);
		System.out.println(coeff[0]);
		System.out.println(coeff[1]);
		System.out.println(coeff[2]);
		System.out.println(coeff[3]);

		
	}
	
	
}
