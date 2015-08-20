function newim = imnormalize(im)
%   newim = imnormalize(im)
%   Normalizes a double image between 0 and 1
%       
%   Julien Lerouge (julien.lerouge@insa-rouen.fr) 13/03/2013
    newim = (im - min(im(:)))/(max(im(:)) - min(im(:)));
end