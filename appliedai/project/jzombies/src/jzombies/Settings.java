package jzombies;

import repast.simphony.engine.environment.RunEnvironment;
import repast.simphony.parameter.Parameters;

public class Settings {
	private Settings instance;
	private Settings() {}
	
	static public int getZombieCount() {
		return RunEnvironment.getInstance().getParameters().getInteger("zombie_count");
	}
	static public int getHumanCount() {
		return RunEnvironment.getInstance().getParameters().getInteger("human_count");
	}
	static public int getIncubationTicks() {
		Integer i = RunEnvironment.getInstance().getParameters().getInteger("incubation_days");
		return 60 * 24 * i;
	}
	static public int getNotContagiousTicks() {
		Integer i = RunEnvironment.getInstance().getParameters().getInteger("not_contagious_days");
		return 60 * 24 * i;
	}
	static public int getMaskPercentage() {
		Integer i = RunEnvironment.getInstance().getParameters().getInteger("mask_percentage");
		return i;
	}
	static public int getDistancingSense() {
		Integer i = RunEnvironment.getInstance().getParameters().getInteger("distancing_sense");
		return i;
	}
	static public int getVaccinationRate() {
		Integer i = RunEnvironment.getInstance().getParameters().getInteger("vaccination_percentage");
		return i;
	}
	
}
