from torch.autograd import Variable
from torch import no_grad as torch_no_grad
from tqdm import tqdm

def train_loop(
    model,
    optimizer,
    data_loader,
    current_epoch,       
    batch_size = 32, 
    cuda = False
    ):
    
    # prepare optimizer and model
    model.train()
    
    train_avg_KLD = 0
    train_avg_repo = 0
    
    size_dataset = len(data_loader.dataset)
    num_batches = len(data_loader)
    data_stream = tqdm(enumerate(data_loader, 1))
    

    for batch_index, (x, _) in data_stream:
        # At what iteration are we?
        iteration = (current_epoch-1)*(len(size_dataset)//batch_size) + batch_index

        # prepare data on gpu if needed
        x = Variable(x).cuda() if cuda else Variable(x)

        # flush gradients and run the model forward
        optimizer.zero_grad()
        (mean, logvar), x_reconstructed = model(x)
        reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss
        
        train_avg_repo += reconstruction_loss.item()
        train_avg_KLD += kl_divergence_loss.item()
        
        # backprop gradients from the loss
        total_loss.backward()
        optimizer.step()

        # update progress
        data_stream.set_description((
            'epoch: {epoch} | '
            'iteration: {iteration} | '
            'progress: [{trained}/{total}] ({progress:.0f}%) | '
            'loss => '
            'total: {total_loss:.4f} / '
            're: {reconstruction_loss:.3f} / '
            'kl: {kl_divergence_loss:.3f}'
        ).format(
            epoch = current_epoch,
            iteration = iteration,
            trained = batch_index * len(x),
            total = len(data_loader.dataset),
            progress = (100. * batch_index / len(data_loader)),
            total_loss = total_loss.item(),
            reconstruction_loss = reconstruction_loss.item(),
            kl_divergence_loss = kl_divergence_loss.item(),
        ))
    
    train_avg_repo /= num_batches
    train_avg_KLD /= num_batches
    
    return train_avg_repo, train_avg_KLD


def test_model(
    model,
    data_loader,
    cuda = False
    ):
    """
    Input:
        model (inheriting nn.Module): the model (neural network) that is
            being tested.
        data_loader (torch.utils.data.Dataloader): dataloader that parses
            data to the model being tested.
        cuda (bool): whether the model is being tested on GPU or not (in
            which case it is being tested on CPU).

    Description:
        Brutally stolen from Alex. Only tests the performance of the model
            does not update weights.

    Output:
        This function outputs the mean, validation KLD- and Reconstruction
            loss.
    """
    
    model.eval()

    num_batches = len(data_loader)
    val_avg_KLD = 0
    val_avg_repo = 0
    
    with torch_no_grad():
        for (x,_) in data_loader:
            
            # prepare data on gpu if needed
            x = Variable(x).cuda() if cuda else Variable(x)

            # Compute loss
            (mean, logvar), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)

            val_avg_repo += reconstruction_loss.item()
            val_avg_KLD += kl_divergence_loss.item()

    val_avg_repo /= num_batches
    val_avg_KLD /= num_batches
    
    return val_avg_repo, val_avg_KLD 