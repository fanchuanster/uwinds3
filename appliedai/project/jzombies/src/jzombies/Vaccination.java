package jzombies;

public class Vaccination {
	private int timeInTicks;
	private double initialProtectionRate;
	
//	https://finance.eastmoney.com/a/202108282069625754.html
//	https://finance.eastmoney.com/a/202108282069625754.html
//  Jan : 94% => Aug : 16 % => everyday ~ -0.2%
//	
//	openair - 4, 1
//	classroom - 14, 4
//  tick: minutes	
	double getProtectionRate(int tick) {
		return initialProtectionRate - ((timeInTicks + tick) / (60 * 24)) * 0.002f;
	}

	public Vaccination(int days, double protectionRate) {
		this.timeInTicks = days * 24 * 60;
		this.initialProtectionRate = protectionRate;
	}
}
