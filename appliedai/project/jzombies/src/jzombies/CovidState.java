package jzombies;

public enum CovidState {
	NONE			(0),	// healthy
	INFECTED		(1),
	CONTAGIOUS		(2),
	STMPTOMATIC		(4);
	
	private final int state;
	
	private CovidState(int state) {
		this.state = state;
	}
}
