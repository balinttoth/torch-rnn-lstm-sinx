require 'nn'
require 'torch'
require 'rnn'
require 'cunn'
require 'gnuplot'
require 'nngraph'

-- constructs data for nn.Sequencer()
function constructData(data, no_param, seq_length, step_size, batch_step_size)
	dim=data:size(1)/batch_step_size
	local result = torch.FloatTensor(dim, no_param*seq_length)
	for k = 1,dim do
		for i = 1,seq_length do
			for j = 1, no_param do
				if (k-1)*batch_step_size + (i-1)*(no_param+step_size-1) + j > data:size(1) then 
					break 
				end
				result[{k,(i-1)*no_param + j}] = data[(k-1)*batch_step_size + (i-1)*(no_param+step_size-1) + j]
			end
		end
	end

	return result
end

no_param = 1 -- number of parameters in one element. In case of x->sin(x) this is 1.
step_size = 1 -- how many steps between sequence elements
seq_length = 20 -- length of the sequence (number of Tensors in the table fed to nn.Sequencer())
batch_step_size = seq_length -- how many steps between batches
batch_size = 10 -- size of the batches

max_epochs =100 -- maximum epochs
lr=0.001 -- learning rate
momentum=0.1 -- momentum

nn.FastLSTM.usenngraph=true

lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.Linear(no_param,256))
      :add(nn.FastLSTM(256,256))
      :add(nn.Linear(256,no_param))
      :add(nn.Tanh())
   )

criterion = nn.SequencerCriterion(nn.MSECriterion())

print(lstm)
print(criterion)

-- the input data is: x[i]=i (i=1..1000)
data = torch.FloatTensor(1000):range(1,1000) --:apply(math.rad):apply(math.sin)*0.8
inp = constructData(data, no_param, seq_length, step_size, batch_step_size)

-- the output data is the sinus of the next timestep: y[i]=sin(x[i+seq_length]) (i=seq_length..1000+seq_length)
data = torch.FloatTensor(1000):range(seq_length,1000+seq_length):apply(math.rad):apply(math.sin)*0.9
oup = constructData(data, no_param, seq_length, step_size, batch_step_size)

-- moving everything to the GPU
lstm:cuda()
criterion:cuda()
inp=inp:cuda()
oup=oup:cuda()

-- setting up training mode

lstm:training()
for epoch=1,max_epochs do
 errsum = 0

 for i = 1,inp:size(1),batch_size do
    if (i+batch_size>inp:size(1)) then 
	break 
    end

    -- get the next batch for inputs and outputs 
    inputs=inp:narrow(1,i,batch_size)
    inputs=inputs:split(no_param,2)
    targets=oup:narrow(1,i,batch_size)
    targets=targets:split(no_param,2) 

    lstm:zeroGradParameters()
    
    local output = lstm:forward(inputs)
    local err = criterion:forward(output, targets)

    local gradOutputs = criterion:backward(output, targets)

    --[[
    -- Sequencer handles the backwardThroughTime internally
    if opt.cutoff > 0 then
       local norm = lstm:gradParamClip(1) --opt.cutoff) -- affects gradParams
       opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
    end
    ]]--

    local gradInputs = lstm:backward(inputs, gradOutputs)
    lstm:updateGradParameters(momentum) -- affects gradParams	    
    lstm:updateParameters(lr) -- affects Params

    -- lstm:forget() -- this doesn't have too much effect in this example

    errsum = errsum + err/(seq_length*batch_size)
  end
  print("Epoch: " .. epoch ..", error: " ..errsum)
end

--- evaluation

lstm:evaluate()

Y=torch.zeros(1000) -- this will contain the predicted output
Y2=Y:clone() -- this is the target output

-- setting up the first input
inputs=inp:narrow(1,1,1):cuda():split(no_param,2)

for i = 1, Y:size(1)/seq_length do
	local y = lstm:forward(inputs)
	
	inputs={}
	-- targets only needed for the figures
	targets=oup:narrow(1,i,1):cuda():split(no_param,2)

	for j=1,seq_length do
		Y[(i-1)*seq_length+j] = y[j][{1,1}] -- predicted
		Y2[(i-1)*seq_length+j] = targets[j]:clone()[{1,1}] -- target
		inputs[j]=y[j]:clone() -- set up the next input
	end
end

gnuplot.pngfigure("output.png")
gnuplot.plot({'predicted',Y:float(),'-'},{'target',Y2:float(),'-'})
gnuplot.plotflush()
