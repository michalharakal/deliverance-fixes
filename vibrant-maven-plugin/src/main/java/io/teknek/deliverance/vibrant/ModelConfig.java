package io.teknek.deliverance.vibrant;

import org.apache.maven.plugins.annotations.Parameter;

public class ModelConfig {
    @Parameter(name="modelName" , defaultValue = "Llama-3.2-3B-Instruct-JQ4")
    private String modelName;

    @Parameter(name= "owner", defaultValue = "tjake")
    private String owner;

    public ModelConfig(){
        setModelName("Llama-3.2-3B-Instruct-JQ4");
        setOwner("tjake");
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getOwner() {
        return owner;
    }

    public void setOwner(String owner) {
        this.owner = owner;
    }

    @Override
    public String toString() {
        return "ModelConfig{" +
                "modelName='" + modelName + '\'' +
                ", owner='" + owner + '\'' +
                '}';
    }
}
