package phrase.jointtopic;

import phrase.jointtopic.EmissionDistribution;

import java.util.Map;

public interface HMM {
    EmissionDistribution getBackground();
    Map<String,EmissionDistribution> getEmissions();
}
