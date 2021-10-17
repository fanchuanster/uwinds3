package jzombies;

import repast.simphony.engine.environment.RunEnvironment;
import repast.simphony.parameter.Parameters;

public class Infection {
//	open air - 4, 1
//	classroom - 14, 4
	
//	Parameters params = RunEnvironment.getInstance ().getParameters ();
//	double naturalInfectionRate = params.getInteger ("natual_infection_rate");

	static double naturalInfectionRate = 0.14f;
	static double meterInfectionRate = 0.04f;
//	
//	private double distance;
//	private boolean indoor;
//	private boolean masked1; 
//	private boolean masked2;
//	private Vaccination vaccination;
//	
//	static public Infection(double distance, boolean masked1, boolean masked2, Vaccination vaccination) {
//		this.distance = distance;
//		this.masked1 = masked1;
//		this.masked2 = masked2;
//		this.vaccination = vaccination;
//	}
	
	static public double getInfectionRate(double distance, boolean masked1, boolean masked2, Vaccination vaccination) {
		double rate = Float.MAX_VALUE;
		if (distance < 1.0f) {
			rate = naturalInfectionRate;
		} else {
			rate = meterInfectionRate;
			// reduced to half for every extra 3 meters
			for (double d = (distance - 1.0f); d > 0.0f; d-=3.0) {
				rate = ((d % 3)/3) * 0.5f * rate;
			}
		}
		if (masked1) {
			rate -= rate/3;
		}
		if (masked2) {
			rate -= rate/3;
		}
		
		if (vaccination != null) {
			int tick = (int) RunEnvironment.getInstance().getCurrentSchedule().getTickCount();
			return rate * (1.0f - vaccination.getProtectionRate(tick));
		} else {
			return rate;
		}		
	}
}
