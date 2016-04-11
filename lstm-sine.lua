require 'nn'
require 'torch'
require 'rnn'
require 'cunn'
require 'gnuplot'
require 'optim'

no_param = 1 -- number of parameters in one element. In case of x->sin(x) this is 1.
step_size = 1 -- how many steps between sequence elements
seq_length = 20 -- length of the sequence (number of Tensors in the table fed to nn.Sequencer())
batch_step_size = seq_length -- how many steps between batches
batch_size = 10 -- size of the batches

max_epochs = 500 -- maximum epochs
lr = 0.01 -- learning rate
mom = 0.9 -- momentum

lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.Linear(no_param,256))
      :add(nn.FastLSTM(256, 256))
      :add(nn.Linear(256, no_param))
      :add(nn.Tanh())
   )

criterion = nn.SequencerCriterion(nn.MSECriterion())

print(lstm)
print(criterion)

-- the input data is: x[i]=i (i=1..1000)
data = torch.Tensor(1000):range(1,1000) --:apply(math.rad):apply(math.sin)*0.8
-- Simpler data creation
--input = data:reshape(20, 50, 1)
input=data:reshape(50,20,1):permute(2,1,3) --:reshape(20,50,1)

-- Normalize the data
input_mean = data:mean()
input_std = data:std()
input:add(-input_mean)
input:div(input_std)

-- the output data is the sinus of the next timestep: y[i]=sin(x[i+seq_length]) (i=seq_length..1000+seq_length)
sinus = torch.Tensor(1000):range(seq_length+1,1000+seq_length):apply(math.rad):apply(math.sin)*0.9
-- Simpler target creation
--target = sinus:reshape(20, 50, 1)
target=sinus:reshape(50,20,1):permute(2,1,3) --:reshape(20,50,1)

-- moving everything to the GPU
lstm:cuda()
criterion:cuda()
input = input:cuda()
target = target:cuda()

-- setting up training mode
lstm:training()

for epoch = 1,max_epochs do
  errsum = 0
  -- Shuffling for better generalized training
  shuffle = torch.randperm(input:size(2))

  for i = 1,input:size(2),batch_size do
    num_samples = math.min(batch_size, input:size(2) - i + 1) 
    inputs = torch.CudaTensor(20, num_samples, 1)
    targets = torch.CudaTensor(20, num_samples, 1)
    
    for k = 1, num_samples do
      inputs[{{}, {k}}] = input[{{}, {shuffle[i+k-1]}}]
      targets[{{}, {k}}] = target[{{}, {shuffle[i+k-1]}}]
    end
    
    -- Since LSTM expects table along time_step dimension
    inputs = nn.SplitTable(1):forward(inputs)
    targets = nn.SplitTable(1):forward(targets)
    lstm:zeroGradParameters()
    
    local outputs = lstm:forward(inputs)
    local err = criterion:forward(outputs, targets)
    local gradOutputs = criterion:backward(outputs, targets)
    local gradInputs = lstm:backward(inputs, gradOutputs)
    lstm:updateGradParameters(mom) -- affects gradParams	    
    lstm:updateParameters(lr) -- affects Params

    errsum = errsum + err / (seq_length * num_samples)

  end
  print("Epoch: " .. epoch ..", error: " ..errsum)
end

-- evaluation
lstm:evaluate()
output = torch.Tensor(20, 50, 1)
num_samples = math.min(batch_size, input:size(2)) 
inputs = input[{{}, {1, num_samples}}] -- feeding the train data as input

for i = 1,input:size(2),batch_size do
  num_samples = math.min(batch_size, input:size(2) - i + 1) 
  --inputs = input[{{}, {i, i+num_samples-1}}] -- feeding the train data as input
  inputs = nn.SplitTable(1):forward(inputs)
  
  local outputs = lstm:forward(inputs)
  -- Since LSTM outputs a table of outputs for each time step, need to combine them
  output[{{}, {i, i+num_samples-1}}] = nn.JoinTable(1):forward(outputs)
  inputs = output[{{}, {i, i+num_samples-1}}]:clone():cuda() -- feeding the output of the neural net as input
end

gnuplot.pngfigure("output_2.png")
gnuplot.plot({'output',output:permute(2,1,3):reshape(1000):float(),'-'}, {'target',sinus,'-'})
gnuplot.plotflush()
