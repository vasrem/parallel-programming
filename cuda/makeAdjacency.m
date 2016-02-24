function A = makeAdjacency(n,p,w)
% make adjacency matrix A(1:n,1:n) where a edge is generated with probability p
% and random edge weights (0:w)
%
% e.g. A = makeAdjacency(10,0.7,30) makes a 10x10 adjacency matrix with
% edge weights 0:30 with 0.7 probability

A = zeros(n);

for i=1:n
  for j=1:n
    if rand()>p
      A(i,j) = inf;
    else
      A(i,j) = rand()*w;
    end
  end
  A(i,i) = 0;
end
save('input.txt','n','A','-ascii')
