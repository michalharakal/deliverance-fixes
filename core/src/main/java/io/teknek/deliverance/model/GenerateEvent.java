package io.teknek.deliverance.model;

public interface GenerateEvent {
    void emit(int next, String nextRaw, String nextCleaned, float timing);
}
